#include <c10/cuda/CUDAGuard.h>
#include <math_constants.h>
#include <torch/extension.h>

#include <cub/cub.cuh>
#include <iostream>

#include "ATen/ATen.h"
#include "ATen/cuda/CUDAContext.h"
#include "compat.h"

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)

__inline__ __device__ float WarpAllReduceMax(float val) {
    for (int mask = 1; mask < 32; mask *= 2) {
        val = max(val, __shfl_xor_sync(0xffffffff, val, mask));
    }
    return val;
}

__inline__ __device__ float WarpAllReduceSum(float val) {
    for (int mask = 1; mask < 32; mask *= 2) {
        val += __shfl_xor_sync(0xffffffff, val, mask);
    }
    return val;
}

inline cudaError_t GetNumBlocks(int64_t block_size, int64_t max_blocks, int64_t waves,
                                int *num_blocks) {
    int dev;
    {
        cudaError_t err = cudaGetDevice(&dev);
        if (err != cudaSuccess) {
            return err;
        }
    }
    int sm_count;
    {
        cudaError_t err = cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev);
        if (err != cudaSuccess) {
            return err;
        }
    }
    int tpm;
    {
        cudaError_t err = cudaDeviceGetAttribute(&tpm, cudaDevAttrMaxThreadsPerMultiProcessor, dev);
        if (err != cudaSuccess) {
            return err;
        }
    }
    *num_blocks =
        std::max<int>(1, std::min<int64_t>(max_blocks, sm_count * tpm / block_size * waves));
    return cudaSuccess;
}

template <typename T>
struct SumOp {
    __device__ __forceinline__ T operator()(const T &a, const T &b) const { return a + b; }
};

template <typename T>
struct MaxOp {
    __device__ __forceinline__ T operator()(const T &a, const T &b) const { return max(a, b); }
};

template <template <typename> class ReductionOp, typename T, int block_size>
__inline__ __device__ T BlockAllReduce(T val) {
    typedef cub::BlockReduce<T, block_size> BlockReduce;
    __shared__ typename BlockReduce::TempStorage temp_storage;
    __shared__ T result_broadcast;
    T result = BlockReduce(temp_storage).Reduce(val, ReductionOp<T>());
    if (threadIdx.x == 0) {
        result_broadcast = result;
    }
    __syncthreads();
    return result_broadcast;
}
////////////////

template <typename T, int cols_per_thread>
__global__ void fastfold_softmax(T *input, T *output, long long rows, long long cols) {
    int threadidx_x = threadIdx.x / 32;
    int threadidx_y = threadIdx.x % 32;
    long long row_offset = (long long)blockIdx.x * 4 + threadidx_x;

    float buf[cols_per_thread];

    int lane_id = threadidx_y;

    if (row_offset < rows) {
        T *row_input = input + row_offset * cols;
        T *row_output = output + row_offset * cols;

        float thread_max = -1 * CUDART_INF_F;

#pragma unroll
        for (int i = 0; i < cols_per_thread; i++) {
            if (lane_id * cols_per_thread + i < cols) {
                buf[i] = static_cast<T>(row_input[lane_id * cols_per_thread + i]);
            } else {
                buf[i] = -1 * CUDART_INF_F;
            }
        }

#pragma unroll
        for (int i = 0; i < cols_per_thread; i++) {
            thread_max = max(thread_max, buf[i]);
        }

        float warp_max = WarpAllReduceMax(thread_max);

        float thread_sum = 0.f;
#pragma unroll
        for (int i = 0; i < cols_per_thread; ++i) {
            buf[i] = __expf(buf[i] - warp_max);
            thread_sum += buf[i];
        }

        float warp_sum = WarpAllReduceSum(thread_sum);
#pragma unroll
        for (int i = 0; i < cols_per_thread; ++i) {
            if (lane_id * cols_per_thread + i < cols) {
                row_output[lane_id * cols_per_thread + i] =
                    static_cast<T>(__fdividef(buf[i], warp_sum));
            }
        }
    }
}

template <typename T, int block_size>
__global__ void fastfold_softmax_sm(T *input, T *output, long long rows, long long cols) {
    extern __shared__ __align__(sizeof(double)) unsigned char shared_buf[];
    auto *buf = reinterpret_cast<float *>(shared_buf);
    const int tid = threadIdx.x;

    for (int64_t row = blockIdx.x; row < rows; row += gridDim.x) {
        float thread_max = -1 * CUDART_INF_F;
        for (int id = tid; id < cols; id += block_size) {
            buf[id] = static_cast<T>(input[row * cols + id]);
            thread_max = max(thread_max, buf[id]);
        }

        const float row_max = BlockAllReduce<MaxOp, float, block_size>(thread_max);

        float thread_sum = 0;
        for (int id = tid; id < cols; id += block_size) {
            buf[id] = __expf(buf[id] - row_max);
            thread_sum += buf[id];
        }

        const float row_sum = BlockAllReduce<SumOp, float, block_size>(thread_sum);

        for (int id = tid; id < cols; id += block_size) {
            output[row * cols + id] = static_cast<T>(buf[id] / row_sum);
        }
    }
}

at::Tensor softmax(at::Tensor input, long long rows, long long cols) {
    CHECK_INPUT(input);
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));

    at::Tensor output = at::empty_like(input);

    int grid = (rows + 3) / 4;
    dim3 block(128);

    if (cols <= 32) {
        if (input.dtype() == torch::kFloat32) {
            fastfold_softmax<float, 1><<<grid, block>>>((float *)input.data_ptr(),
                                                        (float *)output.data_ptr(), rows, cols);
        } else if (input.dtype() == torch::kFloat16) {
            fastfold_softmax<at::Half, 1><<<grid, block>>>(
                (at::Half *)input.data_ptr(), (at::Half *)output.data_ptr(), rows, cols);
        } else if (input.dtype() == torch::kBFloat16) {
            fastfold_softmax<at::BFloat16, 1><<<grid, block>>>(
                (at::BFloat16 *)input.data_ptr(), (at::BFloat16 *)output.data_ptr(), rows, cols);
        }
    }

#define COLS_CASE(col_per_thread)                                                                 \
    else if (cols <= col_per_thread * 32) {                                                       \
        if (input.dtype() == torch::kFloat32) {                                                   \
            fastfold_softmax<float, col_per_thread><<<grid, block>>>(                             \
                (float *)input.data_ptr(), (float *)output.data_ptr(), rows, cols);               \
        } else if (input.dtype() == torch::kFloat16) {                                            \
            fastfold_softmax<at::Half, col_per_thread><<<grid, block>>>(                          \
                (at::Half *)input.data_ptr(), (at::Half *)output.data_ptr(), rows, cols);         \
        } else if (input.dtype() == torch::kBFloat16) {                                           \
            fastfold_softmax<at::BFloat16, col_per_thread><<<grid, block>>>(                      \
                (at::BFloat16 *)input.data_ptr(), (at::BFloat16 *)output.data_ptr(), rows, cols); \
        }                                                                                         \
    }
    COLS_CASE(2)
    COLS_CASE(3)
    COLS_CASE(4)
    COLS_CASE(5)
    COLS_CASE(6)
    COLS_CASE(7)
    COLS_CASE(8)
    COLS_CASE(9)
    COLS_CASE(10)
    COLS_CASE(11)
    COLS_CASE(12)
    COLS_CASE(13)
    COLS_CASE(14)
    COLS_CASE(15)
    COLS_CASE(16)
    COLS_CASE(17)
    COLS_CASE(18)
    COLS_CASE(19)
    COLS_CASE(20)
    COLS_CASE(21)
    COLS_CASE(22)
    COLS_CASE(23)
    COLS_CASE(24)
    COLS_CASE(25)
    COLS_CASE(26)
    COLS_CASE(27)
    COLS_CASE(28)
    COLS_CASE(29)
    COLS_CASE(30)
    COLS_CASE(31)
    COLS_CASE(32)
#undef COLS_CASE
    else {
        int grid_dim;
        constexpr int waves = 32;
        GetNumBlocks(128, rows, waves, &grid_dim);
        dim3 block(128);

        const size_t smem = cols * sizeof(float);

        if (input.dtype() == torch::kFloat32) {
            fastfold_softmax_sm<float, 128><<<grid_dim, block, smem>>>(
                (float *)input.data_ptr(), (float *)output.data_ptr(), rows, cols);
        } else if (input.dtype() == torch::kFloat16) {
            fastfold_softmax_sm<at::Half, 128><<<grid_dim, block, smem>>>(
                (at::Half *)input.data_ptr(), (at::Half *)output.data_ptr(), rows, cols);
        } else if (input.dtype() == torch::kBFloat16) {
            fastfold_softmax_sm<at::BFloat16, 128><<<grid_dim, block, smem>>>(
                (at::BFloat16 *)input.data_ptr(), (at::BFloat16 *)output.data_ptr(), rows, cols);
        }
    }
    return output;
}

template <typename T>
__global__ void fastfold_softmax_grad(T *d_output, T *output, T *d_input, long long rows,
                                      long long cols) {
    int threadidx_x = threadIdx.x / 32;
    int threadidx_y = threadIdx.x % 32;
    long long row_offset = (long long)blockIdx.x * 4 + threadidx_x;
    int cols_per_thread = (cols + 31) / 32;
    int cols_this_thread = cols_per_thread;

    int last_y = (cols / cols_per_thread);

    if (threadidx_y == last_y) {
        cols_this_thread = cols - cols_per_thread * last_y;
    } else if (threadidx_y > last_y) {
        cols_this_thread = 0;
    }

    float y_buf[32];
    float dy_buf[32];

    int lane_id = threadidx_y;

    if (row_offset < rows) {
        T *row_d_output = d_output + row_offset * cols;
        T *row_output = output + row_offset * cols;
        T *row_d_input = d_input + row_offset * cols;

        float thread_max = -1 * CUDART_INF_F;

#pragma unroll
        for (int i = 0; i < cols_this_thread; i++) {
            if (lane_id * cols_per_thread + i < cols) {
                y_buf[i] = static_cast<T>(row_output[lane_id * cols_per_thread + i]);
                dy_buf[i] = static_cast<T>(row_d_output[lane_id * cols_per_thread + i]);
            }
        }

        float thread_sum = 0.f;

#pragma unroll
        for (int i = 0; i < cols_this_thread; i++) {
            if (lane_id * cols_per_thread + i < cols) {
                thread_sum += y_buf[i] * dy_buf[i];
            }
        }

        float warp_sum = WarpAllReduceSum(thread_sum);

#pragma unroll
        for (int i = 0; i < cols_this_thread; ++i) {
            if (lane_id * cols_per_thread + i < cols) {
                row_d_input[lane_id * cols_per_thread + i] =
                    static_cast<T>((dy_buf[i] - warp_sum) * y_buf[i]);
            }
        }
    }
}

at::Tensor softmax_gradient(at::Tensor d_output, at::Tensor output, long long rows,
                            long long cols) {
    CHECK_INPUT(output);
    const at::cuda::OptionalCUDAGuard device_guard(device_of(output));
    at::Tensor grad_input = at::empty_like(output);

    int grid = (rows + 3) / 4;
    dim3 block(128);

    if (output.dtype() == torch::kFloat32) {
        fastfold_softmax_grad<float><<<grid, block>>>((float *)d_output.data_ptr(),
                                                      (float *)output.data_ptr(),
                                                      (float *)grad_input.data_ptr(), rows, cols);
    } else if (output.dtype() == torch::kFloat16) {
        fastfold_softmax_grad<at::Half>
            <<<grid, block>>>((at::Half *)d_output.data_ptr(), (at::Half *)output.data_ptr(),
                              (at::Half *)grad_input.data_ptr(), rows, cols);
    } else if (output.dtype() == torch::kBFloat16) {
        fastfold_softmax_grad<at::BFloat16><<<grid, block>>>(
            (at::BFloat16 *)d_output.data_ptr(), (at::BFloat16 *)output.data_ptr(),
            (at::BFloat16 *)grad_input.data_ptr(), rows, cols);
    }

    return grad_input;
}

////////////////

template <typename T, int cols_per_thread>
__global__ void fastfold_softmax_mask(T *input, T *mask, T *output, long long rows, long long cols,
                                      int head) {
    int threadidx_x = threadIdx.x / 32;
    int threadidx_y = threadIdx.x % 32;
    long long row_offset = (long long)blockIdx.x * 4 + threadidx_x;

    float buf[cols_per_thread];

    int lane_id = threadidx_y;

    T *row_input = input + row_offset * cols;
    T *row_output = output + row_offset * cols;
    T *mask_ptr = mask + ((row_offset / (head * cols)) * cols);

#pragma unroll
    for (int i = 0; i < cols_per_thread; i++) {
        if (lane_id * cols_per_thread + i < cols) {
            if (mask_ptr[lane_id * cols_per_thread + i] == 0) {
                buf[i] = -1 * 1e9;
            } else {
                buf[i] = static_cast<T>(row_input[lane_id * cols_per_thread + i]);
            }
        } else {
            buf[i] = -1 * CUDART_INF_F;
        }
    }

    float thread_max = -1 * CUDART_INF_F;
#pragma unroll
    for (int i = 0; i < cols_per_thread; i++) {
        thread_max = max(thread_max, buf[i]);
    }

    float warp_max = WarpAllReduceMax(thread_max);

    float thread_sum = 0.f;
#pragma unroll
    for (int i = 0; i < cols_per_thread; ++i) {
        buf[i] = __expf(buf[i] - warp_max);
        thread_sum += buf[i];
    }

    float warp_sum = WarpAllReduceSum(thread_sum);
#pragma unroll
    for (int i = 0; i < cols_per_thread; ++i) {
        if (lane_id * cols_per_thread + i < cols) {
            row_output[lane_id * cols_per_thread + i] = static_cast<T>(__fdividef(buf[i], warp_sum));
        }
    }
}

template <typename T, int block_size>
__global__ void fastfold_softmax_mask_sm(T *input, T *mask, T *output, long long rows,
                                         long long cols, int head) {
    extern __shared__ __align__(sizeof(double)) unsigned char shared_buf[];
    auto *buf = reinterpret_cast<float *>(shared_buf);
    const int tid = threadIdx.x;

    for (int64_t row = blockIdx.x; row < rows; row += gridDim.x) {
        T *mask_ptr = mask + ((row / (head * cols)) * cols);
        float thread_max = -1 * CUDART_INF_F;
        for (int id = tid; id < cols; id += block_size) {
            if (mask_ptr[id] == 0) {
                buf[id] = -1 * 1e9;
            } else {
                buf[id] = input[row * cols + id];
            }
            thread_max = max(thread_max, buf[id]);
        }

        const float row_max = BlockAllReduce<MaxOp, float, block_size>(thread_max);

        float thread_sum = 0;
        for (int id = tid; id < cols; id += block_size) {
            buf[id] = __expf(buf[id] - row_max);
            thread_sum += buf[id];
        }

        const float row_sum = BlockAllReduce<SumOp, float, block_size>(thread_sum);

        for (int id = tid; id < cols; id += block_size) {
            output[row * cols + id] = buf[id] / row_sum;
        }
    }
}

at::Tensor fused_mask_softmax_forward(at::Tensor input, at::Tensor mask, long long rows,
                                      long long cols) {
    CHECK_INPUT(input);
    CHECK_INPUT(mask);
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    int head = input.sizes()[2];
    // at::Tensor output = at::empty_like(input);

    int grid = (rows + 3) / 4;
    dim3 block(128);

    if (cols <= 32) {
        if (input.dtype() == torch::kFloat32) {
            fastfold_softmax_mask<float, 1>
                <<<grid, block>>>((float *)input.data_ptr(), (float *)mask.data_ptr(),
                                  (float *)input.data_ptr(), rows, cols, head);
        } else if (input.dtype() == torch::kFloat16) {
            fastfold_softmax_mask<at::Half, 1>
                <<<grid, block>>>((at::Half *)input.data_ptr(), (at::Half *)mask.data_ptr(),
                                  (at::Half *)input.data_ptr(), rows, cols, head);
        } else if (input.dtype() == torch::kBFloat16) {
            fastfold_softmax_mask<at::BFloat16, 1>
                <<<grid, block>>>((at::BFloat16 *)input.data_ptr(), (at::BFloat16 *)mask.data_ptr(),
                                  (at::BFloat16 *)input.data_ptr(), rows, cols, head);
        }
    }
#define COLS_CASE(col_per_thread)                                                            \
    else if (cols <= col_per_thread * 32) {                                                  \
        if (input.dtype() == torch::kFloat32) {                                              \
            fastfold_softmax_mask<float, col_per_thread>                                     \
                <<<grid, block>>>((float *)input.data_ptr(), (float *)mask.data_ptr(),       \
                                  (float *)input.data_ptr(), rows, cols, head);              \
        } else if (input.dtype() == torch::kFloat16) {                                       \
            fastfold_softmax_mask<at::Half, col_per_thread>                                  \
                <<<grid, block>>>((at::Half *)input.data_ptr(), (at::Half *)mask.data_ptr(), \
                                  (at::Half *)input.data_ptr(), rows, cols, head);           \
        } else if (input.dtype() == torch::kBFloat16) {                                      \
            fastfold_softmax_mask<at::BFloat16, col_per_thread><<<grid, block>>>(            \
                (at::BFloat16 *)input.data_ptr(), (at::BFloat16 *)mask.data_ptr(),           \
                (at::BFloat16 *)input.data_ptr(), rows, cols, head);                         \
        }                                                                                    \
    }
    COLS_CASE(2)
    COLS_CASE(3)
    COLS_CASE(4)
    COLS_CASE(5)
    COLS_CASE(6)
    COLS_CASE(7)
    COLS_CASE(8)
    COLS_CASE(9)
    COLS_CASE(10)
    COLS_CASE(11)
    COLS_CASE(12)
    COLS_CASE(13)
    COLS_CASE(14)
    COLS_CASE(15)
    COLS_CASE(16)
    COLS_CASE(17)
    COLS_CASE(18)
    COLS_CASE(19)
    COLS_CASE(20)
    COLS_CASE(21)
    COLS_CASE(22)
    COLS_CASE(23)
    COLS_CASE(24)
    COLS_CASE(25)
    COLS_CASE(26)
    COLS_CASE(27)
    COLS_CASE(28)
    COLS_CASE(29)
    COLS_CASE(30)
    COLS_CASE(31)
    COLS_CASE(32)
#undef COLS_CASE
    else {
        int grid_dim;
        constexpr int waves = 32;
        GetNumBlocks(128, rows, waves, &grid_dim);
        dim3 block(128);

        const size_t smem = cols * sizeof(float);

        if (input.dtype() == torch::kFloat32) {
            fastfold_softmax_mask_sm<float, 128>
                <<<grid, block, smem>>>((float *)input.data_ptr(), (float *)mask.data_ptr(),
                                        (float *)input.data_ptr(), rows, cols, head);
        } else if (input.dtype() == torch::kFloat16) {
            fastfold_softmax_mask_sm<at::Half, 128>
                <<<grid, block, smem>>>((at::Half *)input.data_ptr(), (at::Half *)mask.data_ptr(),
                                        (at::Half *)input.data_ptr(), rows, cols, head);
        } else if (input.dtype() == torch::kBFloat16) {
            fastfold_softmax_mask_sm<at::BFloat16, 128><<<grid, block, smem>>>(
                (at::BFloat16 *)input.data_ptr(), (at::BFloat16 *)mask.data_ptr(),
                (at::BFloat16 *)input.data_ptr(), rows, cols, head);
        }
    }
    return input;
}

template <typename T>
__global__ void fastfold_softmax_mask_grad(T *d_output, T *output, T *d_input, T *mask,
                                           long long rows, long long cols, int head) {
    int threadidx_x = threadIdx.x / 32;
    int threadidx_y = threadIdx.x % 32;
    long long row_offset = (long long)blockIdx.x * 4 + threadidx_x;
    int cols_per_thread = (cols + 31) / 32;
    int cols_this_thread = cols_per_thread;

    int last_y = (cols / cols_per_thread);

    if (threadidx_y == last_y) {
        cols_this_thread = cols - cols_per_thread * last_y;
    } else if (threadidx_y > last_y) {
        cols_this_thread = 0;
    }

    float y_buf[32];
    float dy_buf[32];

    int lane_id = threadidx_y;

    if (row_offset < rows) {
        T *row_d_output = d_output + row_offset * cols;
        T *row_output = output + row_offset * cols;
        T *row_d_input = d_input + row_offset * cols;
        T *mask_ptr = mask + ((row_offset / (head * cols)) * cols);

        float thread_max = -1 * CUDART_INF_F;

#pragma unroll
        for (int i = 0; i < cols_this_thread; i++) {
            if (lane_id * cols_per_thread + i < cols) {
                y_buf[i] = static_cast<T>(row_output[lane_id * cols_per_thread + i]);
                dy_buf[i] = static_cast<T>(row_d_output[lane_id * cols_per_thread + i]);
            }
        }

        float thread_sum = 0.f;

#pragma unroll
        for (int i = 0; i < cols_this_thread; i++) {
            if (lane_id * cols_per_thread + i < cols) {
                thread_sum += y_buf[i] * dy_buf[i];
            }
        }

        float warp_sum = WarpAllReduceSum(thread_sum);

#pragma unroll
        for (int i = 0; i < cols_this_thread; ++i) {
            if (lane_id * cols_per_thread + i < cols) {
                if (mask_ptr[lane_id * cols_per_thread + i] != 0) {
                    row_d_input[lane_id * cols_per_thread + i] =
                        static_cast<T>((dy_buf[i] - warp_sum) * y_buf[i]);
                } else {
                    row_d_input[lane_id * cols_per_thread + i] = 0;
                }
            }
        }
    }
}

at::Tensor fused_mask_softmax_backward(at::Tensor d_output, at::Tensor output, at::Tensor mask,
                                       long long rows, long long cols) {
    CHECK_INPUT(output);
    CHECK_INPUT(mask);
    const at::cuda::OptionalCUDAGuard device_guard(device_of(mask));
    int head = output.sizes()[2];
    at::Tensor grad_input = at::empty_like(output);

    int grid = (rows + 3) / 4;
    dim3 block(128);

    if (output.dtype() == torch::kFloat32) {
        fastfold_softmax_mask_grad<float><<<grid, block>>>(
            (float *)d_output.data_ptr(), (float *)output.data_ptr(),
            (float *)grad_input.data_ptr(), (float *)mask.data_ptr(), rows, cols, head);
    } else if (output.dtype() == torch::kFloat16) {
        fastfold_softmax_mask_grad<at::Half><<<grid, block>>>(
            (at::Half *)d_output.data_ptr(), (at::Half *)output.data_ptr(),
            (at::Half *)grad_input.data_ptr(), (at::Half *)mask.data_ptr(), rows, cols, head);
    } else if (output.dtype() == torch::kBFloat16) {
        fastfold_softmax_mask_grad<at::BFloat16><<<grid, block>>>(
            (at::BFloat16 *)d_output.data_ptr(), (at::BFloat16 *)output.data_ptr(),
            (at::BFloat16 *)grad_input.data_ptr(), (at::BFloat16 *)mask.data_ptr(), rows, cols,
            head);
    }

    return grad_input;
}

////////////////

template <typename T, int cols_per_thread>
__global__ void fastfold_softmax_mask_bias(T *input, T *mask, T *bias, T *output, long long rows,
                                           long long cols, int head) {
    int threadidx_x = threadIdx.x / 32;
    int threadidx_y = threadIdx.x % 32;
    long long row_offset = (long long)blockIdx.x * 4 + threadidx_x;

    float buf[cols_per_thread];

    int lane_id = threadidx_y;

    T *row_input = input + row_offset * cols;
    T *row_output = output + row_offset * cols;
    T *mask_ptr = mask + ((row_offset / (head * cols)) * cols);
    T *bias_ptr = bias + ((row_offset % (head * cols)) * cols);

#pragma unroll
    for (int i = 0; i < cols_per_thread; i++) {
        if (lane_id * cols_per_thread + i < cols) {
            if (mask_ptr[lane_id * cols_per_thread + i] == 0) {
                buf[i] = -1 * 10e9;
            } else {
                buf[i] = static_cast<T>(row_input[lane_id * cols_per_thread + i]) +
                        static_cast<T>(bias_ptr[lane_id * cols_per_thread + i]);
            }
        } else {
            buf[i] = -1 * CUDART_INF_F;
        }
    }

    float thread_max = -1 * CUDART_INF_F;
#pragma unroll
    for (int i = 0; i < cols_per_thread; i++) {
        thread_max = max(thread_max, buf[i]);
    }

    float warp_max = WarpAllReduceMax(thread_max);

    float thread_sum = 0.f;
#pragma unroll
    for (int i = 0; i < cols_per_thread; ++i) {
        buf[i] = __expf(buf[i] - warp_max);
        thread_sum += buf[i];
    }

    float warp_sum = WarpAllReduceSum(thread_sum);
#pragma unroll
    for (int i = 0; i < cols_per_thread; ++i) {
        if (lane_id * cols_per_thread + i < cols) {
            row_output[lane_id * cols_per_thread + i] = static_cast<T>(__fdividef(buf[i], warp_sum));
        }
    }
}

template <typename T, int block_size>
__global__ void fastfold_softmax_mask_bias_sm(T *input, T *mask, T *bias, T *output, long long rows,
                                              long long cols, int head) {
    extern __shared__ __align__(sizeof(double)) unsigned char shared_buf[];
    auto *buf = reinterpret_cast<float *>(shared_buf);
    const int tid = threadIdx.x;

    for (int64_t row = blockIdx.x; row < rows; row += gridDim.x) {
        T *mask_ptr = mask + ((row / (head * cols)) * cols);
        T *bias_ptr = bias + ((row % (head * cols)) * cols);
        float thread_max = -1 * CUDART_INF_F;
        for (int id = tid; id < cols; id += block_size) {
            if (mask_ptr[id] == 0) {
                buf[id] = -1 * 1e9;
            } else {
                buf[id] = input[row * cols + id] + bias_ptr[id];
            }
            thread_max = max(thread_max, buf[id]);
        }

        const float row_max = BlockAllReduce<MaxOp, float, block_size>(thread_max);

        float thread_sum = 0;
        for (int id = tid; id < cols; id += block_size) {
            buf[id] = __expf(buf[id] - row_max);
            thread_sum += buf[id];
        }

        const float row_sum = BlockAllReduce<SumOp, float, block_size>(thread_sum);

        for (int id = tid; id < cols; id += block_size) {
            output[row * cols + id] = buf[id] / row_sum;
        }
    }
}

at::Tensor fused_mask_bias_softmax_forward(at::Tensor input, at::Tensor mask, at::Tensor bias,
                                           long long rows, long long cols) {
    CHECK_INPUT(input);
    CHECK_INPUT(mask);
    CHECK_INPUT(bias);
    const at::cuda::OptionalCUDAGuard device_guard(device_of(input));
    int head = input.sizes()[2];
    // at::Tensor output = at::empty_like(input);

    int grid = (rows + 3) / 4;
    dim3 block(128);

    if (cols <= 32) {
        if (input.dtype() == torch::kFloat32) {
            fastfold_softmax_mask_bias<float, 1><<<grid, block>>>(
                (float *)input.data_ptr(), (float *)mask.data_ptr(), (float *)bias.data_ptr(),
                (float *)input.data_ptr(), rows, cols, head);
        } else if (input.dtype() == torch::kFloat16) {
            fastfold_softmax_mask_bias<at::Half, 1><<<grid, block>>>(
                (at::Half *)input.data_ptr(), (at::Half *)mask.data_ptr(),
                (at::Half *)bias.data_ptr(), (at::Half *)input.data_ptr(), rows, cols, head);
        } else if (input.dtype() == torch::kBFloat16) {
            fastfold_softmax_mask_bias<at::BFloat16, 1>
                <<<grid, block>>>((at::BFloat16 *)input.data_ptr(), (at::BFloat16 *)mask.data_ptr(),
                                  (at::BFloat16 *)bias.data_ptr(), (at::BFloat16 *)input.data_ptr(),
                                  rows, cols, head);
        }
    }
#define COLS_CASE(col_per_thread)                                                              \
    else if (cols <= col_per_thread * 32) {                                                    \
        if (input.dtype() == torch::kFloat32) {                                                \
            fastfold_softmax_mask_bias<float, col_per_thread><<<grid, block>>>(                \
                (float *)input.data_ptr(), (float *)mask.data_ptr(), (float *)bias.data_ptr(), \
                (float *)input.data_ptr(), rows, cols, head);                                  \
        } else if (input.dtype() == torch::kFloat16) {                                         \
            fastfold_softmax_mask_bias<at::Half, col_per_thread><<<grid, block>>>(             \
                (at::Half *)input.data_ptr(), (at::Half *)mask.data_ptr(),                     \
                (at::Half *)bias.data_ptr(), (at::Half *)input.data_ptr(), rows, cols, head);  \
        } else if (input.dtype() == torch::kBFloat16) {                                        \
            fastfold_softmax_mask_bias<at::BFloat16, col_per_thread><<<grid, block>>>(         \
                (at::BFloat16 *)input.data_ptr(), (at::BFloat16 *)mask.data_ptr(),             \
                (at::BFloat16 *)bias.data_ptr(), (at::BFloat16 *)input.data_ptr(), rows, cols, \
                head);                                                                         \
        }                                                                                      \
    }
    COLS_CASE(2)
    COLS_CASE(3)
    COLS_CASE(4)
    COLS_CASE(5)
    COLS_CASE(6)
    COLS_CASE(7)
    COLS_CASE(8)
    COLS_CASE(9)
    COLS_CASE(10)
    COLS_CASE(11)
    COLS_CASE(12)
    COLS_CASE(13)
    COLS_CASE(14)
    COLS_CASE(15)
    COLS_CASE(16)
    COLS_CASE(17)
    COLS_CASE(18)
    COLS_CASE(19)
    COLS_CASE(20)
    COLS_CASE(21)
    COLS_CASE(22)
    COLS_CASE(23)
    COLS_CASE(24)
    COLS_CASE(25)
    COLS_CASE(26)
    COLS_CASE(27)
    COLS_CASE(28)
    COLS_CASE(29)
    COLS_CASE(30)
    COLS_CASE(31)
    COLS_CASE(32)
#undef COLS_CASE
    else {
        int grid_dim;
        constexpr int waves = 32;
        GetNumBlocks(128, rows, waves, &grid_dim);
        dim3 block(128);

        const size_t smem = cols * sizeof(float);

        if (input.dtype() == torch::kFloat32) {
            fastfold_softmax_mask_bias_sm<float, 128><<<grid, block, smem>>>(
                (float *)input.data_ptr(), (float *)mask.data_ptr(), (float *)bias.data_ptr(),
                (float *)input.data_ptr(), rows, cols, head);
        } else if (input.dtype() == torch::kFloat16) {
            fastfold_softmax_mask_bias_sm<at::Half, 128><<<grid, block, smem>>>(
                (at::Half *)input.data_ptr(), (at::Half *)mask.data_ptr(),
                (at::Half *)bias.data_ptr(), (at::Half *)input.data_ptr(), rows, cols, head);
        } else if (input.dtype() == torch::kBFloat16) {
            fastfold_softmax_mask_bias_sm<at::BFloat16, 128><<<grid, block, smem>>>(
                (at::BFloat16 *)input.data_ptr(), (at::BFloat16 *)mask.data_ptr(),
                (at::BFloat16 *)bias.data_ptr(), (at::BFloat16 *)input.data_ptr(), rows, cols,
                head);
        }
    }

    return input;
}

at::Tensor fused_mask_bias_softmax_backward(at::Tensor d_output, at::Tensor output, at::Tensor mask,
                                            at::Tensor bias, long long rows, long long cols) {
    CHECK_INPUT(output);
    CHECK_INPUT(mask);
    const at::cuda::OptionalCUDAGuard device_guard(device_of(mask));
    int head = output.sizes()[2];
    at::Tensor grad_input = at::empty_like(output);

    int grid = (rows + 3) / 4;
    dim3 block(128);

    if (output.dtype() == torch::kFloat32) {
        fastfold_softmax_mask_grad<float><<<grid, block>>>(
            (float *)d_output.data_ptr(), (float *)output.data_ptr(),
            (float *)grad_input.data_ptr(), (float *)mask.data_ptr(), rows, cols, head);
    } else if (output.dtype() == torch::kFloat16) {
        fastfold_softmax_mask_grad<at::Half><<<grid, block>>>(
            (at::Half *)d_output.data_ptr(), (at::Half *)output.data_ptr(),
            (at::Half *)grad_input.data_ptr(), (at::Half *)mask.data_ptr(), rows, cols, head);
    } else if (output.dtype() == torch::kBFloat16) {
        fastfold_softmax_mask_grad<at::BFloat16><<<grid, block>>>(
            (at::BFloat16 *)d_output.data_ptr(), (at::BFloat16 *)output.data_ptr(),
            (at::BFloat16 *)grad_input.data_ptr(), (at::BFloat16 *)mask.data_ptr(), rows, cols,
            head);
    }

    return grad_input;
}
