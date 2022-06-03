// part of code modified from https://github.com/NVIDIA/apex

#include <cooperative_groups.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

#include <THC/THCDeviceUtils.cuh>

#include "ATen/ATen.h"
#include "ATen/AccumulateType.h"
#include "ATen/cuda/CUDAContext.h"
#include "compat.h"
#include "type_shim.h"

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)

inline __device__ void WelfordOnline(float val, float* mean, float* m2, float* count) {
    *count += 1;
    float delta1 = val - *mean;
    *mean += delta1 / (*count);
    float delta2 = val - *mean;
    *m2 += delta1 * delta2;
}

inline __device__ void WelfordOnline(float b_mean, float b_m2, float b_count, float* mean,
                                     float* m2, float* count) {
    if (b_count == 0) {
        return;
    }
    float new_count = *count + b_count;
    float nb_n = b_count / new_count;
    float delta = b_mean - *mean;
    *mean += delta * nb_n;
    *m2 += b_m2 + delta * delta * (*count) * nb_n;
    *count = new_count;
}

__inline__ __device__ void WelfordWarpAllReduce(float thread_mean, float thread_m2,
                                                float thread_count, float* mean, float* m2,
                                                float* count) {
    *mean = thread_mean;
    *m2 = thread_m2;
    *count = thread_count;
    for (int mask = 1; mask < 32; mask *= 2) {
        float b_mean = __shfl_down_sync(0xffffffff, *mean, mask);
        float b_m2 = __shfl_down_sync(0xffffffff, *m2, mask);
        float b_count = __shfl_down_sync(0xffffffff, *count, mask);
        WelfordOnline(b_mean, b_m2, b_count, mean, m2, count);
    }

    *mean = __shfl_sync(0xffffffff, *mean, 0, 32);
    *m2 = __shfl_sync(0xffffffff, *m2, 0, 32);
    *count = __shfl_sync(0xffffffff, *count, 0, 32);
}

template <typename T>
__global__ void fastfold_layernorm(T* input, T* output, T* gamma, T* beta, float* mean,
                                   float* invvar, int rows, int cols, double epsilon) {
    int threadidx_x = threadIdx.x / 32;
    int threadidx_y = threadIdx.x % 32;
    int row_offset = blockIdx.x * 4 + threadidx_x;
    int cols_per_thread = (cols + 31) / 32;
    int cols_this_thread = cols_per_thread;

    int last_y = (cols / cols_per_thread);

    if (threadidx_y == last_y) {
        cols_this_thread = cols - cols_per_thread * last_y;
    } else if (threadidx_y > last_y) {
        cols_this_thread = 0;
    }

    int lane_id = threadidx_y;

    if (row_offset < rows) {
        float buf[32];

        float thread_mean = 0.f;
        float thread_m2 = 0.f;
        float thread_count = 0.f;

        float warp_mean;
        float warp_m2;
        float warp_count;

        T* row_input = input + row_offset * cols;
        T* row_output = output + row_offset * cols;

#pragma unroll
        for (int i = 0; i < cols_this_thread; i++) {
            buf[i] = static_cast<float>(row_input[lane_id * cols_per_thread + i]);
        }

#pragma unroll
        for (int i = 0; i < cols_this_thread; i++) {
            WelfordOnline(buf[i], &thread_mean, &thread_m2, &thread_count);
        }

        WelfordWarpAllReduce(thread_mean, thread_m2, thread_count, &warp_mean, &warp_m2,
                             &warp_count);

        float row_mean = warp_mean;
        float row_variance = max(warp_m2 / warp_count, 0.f);
        float row_inv_var = rsqrt(row_variance + epsilon);

        if (lane_id == 0) {
            mean[row_offset] = row_mean;
            invvar[row_offset] = row_inv_var;
        }

#pragma unroll
        for (int i = 0; i < cols_this_thread; ++i) {
            buf[i] = (buf[i] - row_mean) * row_inv_var;
        }

#pragma unroll
        for (int i = 0; i < cols_this_thread; ++i) {
            row_output[lane_id * cols_per_thread + i] =
                static_cast<T>(buf[i]) * gamma[lane_id * cols_per_thread + i] +
                beta[lane_id * cols_per_thread + i];
        }
    }
}

void cuda_layer_norm(at::Tensor* output, at::Tensor* mean, at::Tensor* invvar, at::Tensor* input,
                     int rows, int cols, at::IntArrayRef normalized_shape, at::Tensor* gamma,
                     at::Tensor* beta, double epsilon) {
    int grid = (rows + 3) / 4;
    dim3 block(128);

    if (output->dtype() == torch::kFloat32) {
        fastfold_layernorm<float><<<grid, block>>>(
            (float*)input->data_ptr(), (float*)output->data_ptr(), (float*)gamma->data_ptr(),
            (float*)beta->data_ptr(), (float*)mean->data_ptr(), (float*)invvar->data_ptr(), rows,
            cols, epsilon);
    } else if (output->dtype() == torch::kFloat16) {
        fastfold_layernorm<at::Half><<<grid, block>>>(
            (at::Half*)input->data_ptr(), (at::Half*)output->data_ptr(),
            (at::Half*)gamma->data_ptr(), (at::Half*)beta->data_ptr(), (float*)mean->data_ptr(),
            (float*)invvar->data_ptr(), rows, cols, epsilon);
    } else if (output->dtype() == torch::kBFloat16) {
        fastfold_layernorm<at::BFloat16><<<grid, block>>>(
            (at::BFloat16*)input->data_ptr(), (at::BFloat16*)output->data_ptr(),
            (at::BFloat16*)gamma->data_ptr(), (at::BFloat16*)beta->data_ptr(),
            (float*)mean->data_ptr(), (float*)invvar->data_ptr(), rows, cols, epsilon);
    }
}

template <typename T>
struct SharedMemory;

template <>
struct SharedMemory<float> {
    __device__ float* getPointer() {
        extern __shared__ float s_float[];
        return s_float;
    }
};

template <typename T, typename U, typename V>
__device__ void cuLoadWriteStridedInputs(const int i1_block, const int thr_load_row_off,
                                         const int thr_load_col_off, const int i2_off,
                                         const int row_stride, U* warp_buf1, U* warp_buf2,
                                         const T* input, const V* dout, const int i1_end,
                                         const int n2, const U* __restrict__ mean,
                                         const U* __restrict__ invvar) {
    int i1 = i1_block + thr_load_row_off;
    if (i1 < i1_end) {
        U curr_mean = mean[i1];
        U curr_invvar = invvar[i1];
        for (int k = 0; k < blockDim.y; ++k) {
            int i2 = i2_off + k;
            int load_idx = i1 * n2 + i2;
            int write_idx = thr_load_row_off * row_stride + thr_load_col_off + k;
            if (i2 < n2) {
                U curr_input = static_cast<U>(input[load_idx]);
                U curr_dout = static_cast<U>(dout[load_idx]);
                warp_buf1[write_idx] = curr_dout;
                warp_buf2[write_idx] = curr_dout * (curr_input - curr_mean) * curr_invvar;
            } else {
                warp_buf1[write_idx] = U(0);
                warp_buf2[write_idx] = U(0);
            }
        }
    } else {
        for (int k = 0; k < blockDim.y; ++k) {
            int write_idx = thr_load_row_off * row_stride + thr_load_col_off + k;
            warp_buf1[write_idx] = U(0);
            warp_buf2[write_idx] = U(0);
        }
    }
}

template <typename T, typename U, typename V>
__device__ void cuLoadAddStridedInputs(const int i1_block, const int thr_load_row_off,
                                       const int thr_load_col_off, const int i2_off,
                                       const int row_stride, U* warp_buf1, U* warp_buf2,
                                       const T* input, const V* dout, const int i1_end,
                                       const int n2, const U* __restrict__ mean,
                                       const U* __restrict__ invvar) {
    int i1 = i1_block + thr_load_row_off;
    if (i1 < i1_end) {
        U curr_mean = mean[i1];
        U curr_invvar = invvar[i1];
        for (int k = 0; k < blockDim.y; ++k) {
            int i2 = i2_off + k;
            int load_idx = i1 * n2 + i2;
            int write_idx = thr_load_row_off * row_stride + thr_load_col_off + k;
            if (i2 < n2) {
                U curr_input = static_cast<U>(input[load_idx]);
                U curr_dout = static_cast<U>(dout[load_idx]);
                warp_buf1[write_idx] += curr_dout;
                warp_buf2[write_idx] += curr_dout * (curr_input - curr_mean) * curr_invvar;
            }
        }
    }
}

template <typename T, typename U, typename V>
__global__ void cuComputePartGradGammaBeta(const V* __restrict__ dout, const T* __restrict__ input,
                                           const int n1, const int n2, const U* __restrict__ mean,
                                           const U* __restrict__ invvar, U epsilon,
                                           U* part_grad_gamma, U* part_grad_beta) {
    const int numsegs_n1 = (n1 + blockDim.y * blockDim.y - 1) / (blockDim.y * blockDim.y);
    const int segs_per_block = (numsegs_n1 + gridDim.y - 1) / gridDim.y;
    const int i1_beg = blockIdx.y * segs_per_block * blockDim.y * blockDim.y;
    const int i1_beg_plus_one = (blockIdx.y + 1) * segs_per_block * blockDim.y * blockDim.y;
    const int i1_end = i1_beg_plus_one < n1 ? i1_beg_plus_one : n1;
    const int row_stride = blockDim.x + 1;
    const int thr_load_col_off = (threadIdx.x * blockDim.y) & (blockDim.x - 1);
    const int thr_load_row_off = (threadIdx.x * blockDim.y) / blockDim.x + threadIdx.y * blockDim.y;
    const int i2_off = blockIdx.x * blockDim.x + thr_load_col_off;
    SharedMemory<U> shared;
    U* buf = shared.getPointer();  // buf has at least blockDim.x * blockDim.y * blockDim.y +
                                   // (blockDim.y - 1)*(blockDim.x/blockDim.y) elements
    U* warp_buf1 = (U*)buf;
    U* warp_buf2 = warp_buf1 + blockDim.y * blockDim.y * row_stride;
    // compute partial sums from strided inputs
    // do this to increase number of loads in flight
    cuLoadWriteStridedInputs(i1_beg, thr_load_row_off, thr_load_col_off, i2_off, row_stride,
                             warp_buf1, warp_buf2, input, dout, i1_end, n2, mean, invvar);
    for (int i1_block = i1_beg + blockDim.y * blockDim.y; i1_block < i1_end;
         i1_block += blockDim.y * blockDim.y) {
        cuLoadAddStridedInputs(i1_block, thr_load_row_off, thr_load_col_off, i2_off, row_stride,
                               warp_buf1, warp_buf2, input, dout, i1_end, n2, mean, invvar);
    }
    __syncthreads();
    // inter-warp reductions
    // sum within each warp
    U acc1 = U(0);
    U acc2 = U(0);
    for (int k = 0; k < blockDim.y; ++k) {
        int row1 = threadIdx.y + k * blockDim.y;
        int idx1 = row1 * row_stride + threadIdx.x;
        acc1 += warp_buf1[idx1];
        acc2 += warp_buf2[idx1];
    }
    warp_buf1[threadIdx.y * row_stride + threadIdx.x] = acc1;
    warp_buf2[threadIdx.y * row_stride + threadIdx.x] = acc2;
    __syncthreads();
    // sum all warps
    for (int offset = blockDim.y / 2; offset > 1; offset /= 2) {
        if (threadIdx.y < offset) {
            int row1 = threadIdx.y;
            int row2 = threadIdx.y + offset;
            int idx1 = row1 * row_stride + threadIdx.x;
            int idx2 = row2 * row_stride + threadIdx.x;
            warp_buf1[idx1] += warp_buf1[idx2];
            warp_buf2[idx1] += warp_buf2[idx2];
        }
        __syncthreads();
    }
    int i2 = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadIdx.y == 0 && i2 < n2) {
        int row1 = threadIdx.y;
        int row2 = threadIdx.y + 1;
        int idx1 = row1 * row_stride + threadIdx.x;
        int idx2 = row2 * row_stride + threadIdx.x;
        part_grad_beta[blockIdx.y * n2 + i2] = warp_buf1[idx1] + warp_buf1[idx2];
        part_grad_gamma[blockIdx.y * n2 + i2] = warp_buf2[idx1] + warp_buf2[idx2];
    }
}

template <typename U, typename V>
__global__ void cuComputeGradGammaBeta(const U* part_grad_gamma, const U* part_grad_beta,
                                       const int part_size, const int n1, const int n2,
                                       V* grad_gamma, V* grad_beta) {
    // sum partial gradients for gamma and beta
    SharedMemory<U> shared;
    U* buf = shared.getPointer();
    int i2 = blockIdx.x * blockDim.x + threadIdx.x;
    if (i2 < n2) {
        // each warp does sequential reductions until reduced part_size is num_warps
        int num_warp_reductions = part_size / blockDim.y;
        U sum_gamma = U(0);
        U sum_beta = U(0);
        const U* part_grad_gamma_ptr =
            part_grad_gamma + threadIdx.y * num_warp_reductions * n2 + i2;
        const U* part_grad_beta_ptr = part_grad_beta + threadIdx.y * num_warp_reductions * n2 + i2;
        for (int warp_offset = 0; warp_offset < num_warp_reductions; ++warp_offset) {
            sum_gamma += part_grad_gamma_ptr[warp_offset * n2];
            sum_beta += part_grad_beta_ptr[warp_offset * n2];
        }
        // inter-warp reductions
        const int nbsize3 = blockDim.x * blockDim.y / 2;
        for (int offset = blockDim.y / 2; offset >= 1; offset /= 2) {
            // top half write to shared memory
            if (threadIdx.y >= offset && threadIdx.y < 2 * offset) {
                const int write_idx = (threadIdx.y - offset) * blockDim.x + threadIdx.x;
                buf[write_idx] = sum_gamma;
                buf[write_idx + nbsize3] = sum_beta;
            }
            __syncthreads();
            // bottom half sums
            if (threadIdx.y < offset) {
                const int read_idx = threadIdx.y * blockDim.x + threadIdx.x;
                sum_gamma += buf[read_idx];
                sum_beta += buf[read_idx + nbsize3];
            }
            __syncthreads();
        }
        // write out fully summed gradients
        if (threadIdx.y == 0) {
            grad_gamma[i2] = sum_gamma;
            grad_beta[i2] = sum_beta;
        }
    }
}

template <typename T, typename U, typename V>
__global__ void cuComputeGradInput(const V* __restrict__ dout, const T* __restrict__ input,
                                   const int n1, const int n2, const U* __restrict__ mean,
                                   const U* __restrict__ invvar, U epsilon, const V* gamma,
                                   T* grad_input) {
    for (auto i1 = blockIdx.y; i1 < n1; i1 += gridDim.y) {
        U sum_loss1 = U(0);
        U sum_loss2 = U(0);
        const U c_mean = mean[i1];
        const U c_invvar = invvar[i1];
        const T* k_input = input + i1 * n2;
        const V* k_dout = dout + i1 * n2;
        const int numx = blockDim.x * blockDim.y;
        const int thrx = threadIdx.x + threadIdx.y * blockDim.x;
        if (gamma != NULL) {
            int l = 4 * thrx;
            for (; l + 3 < n2; l += 4 * numx) {
                for (int k = 0; k < 4; ++k) {
                    const U c_h = static_cast<U>(k_input[l + k]);
                    const U c_loss = static_cast<U>(k_dout[l + k]);
                    sum_loss1 += c_loss * gamma[l + k];
                    sum_loss2 += c_loss * gamma[l + k] * (c_h - c_mean) * c_invvar;
                }
            }
            for (; l < n2; ++l) {
                const U c_h = static_cast<U>(k_input[l]);
                const U c_loss = static_cast<U>(k_dout[l]);
                sum_loss1 += c_loss * gamma[l];
                sum_loss2 += c_loss * gamma[l] * (c_h - c_mean) * c_invvar;
            }
        } else {
            int l = 4 * thrx;
            for (; l + 3 < n2; l += 4 * numx) {
                for (int k = 0; k < 4; ++k) {
                    const U c_h = static_cast<U>(k_input[l + k]);
                    const U c_loss = static_cast<U>(k_dout[l + k]);
                    sum_loss1 += c_loss;
                    sum_loss2 += c_loss * (c_h - c_mean) * c_invvar;
                }
            }
            for (; l < n2; ++l) {
                const U c_h = static_cast<U>(k_input[l]);
                const U c_loss = static_cast<U>(k_dout[l]);
                sum_loss1 += c_loss;
                sum_loss2 += c_loss * (c_h - c_mean) * c_invvar;
            }
        }
        // intra-warp reductions
        for (int mask = blockDim.x / 2; mask > 0; mask /= 2) {
            sum_loss1 += WARP_SHFL_XOR(sum_loss1, mask);
            sum_loss2 += WARP_SHFL_XOR(sum_loss2, mask);
        }
        // inter-warp reductions
        if (blockDim.y > 1) {
            SharedMemory<U> shared;
            U* buf = shared.getPointer();
            for (int offset = blockDim.y / 2; offset > 0; offset /= 2) {
                // upper half of warps write to shared
                if (threadIdx.y >= offset && threadIdx.y < 2 * offset) {
                    const int wrt_i = (threadIdx.y - offset) * blockDim.x + threadIdx.x;
                    buf[2 * wrt_i] = sum_loss1;
                    buf[2 * wrt_i + 1] = sum_loss2;
                }
                __syncthreads();
                // lower half merges
                if (threadIdx.y < offset) {
                    const int read_i = threadIdx.y * blockDim.x + threadIdx.x;
                    sum_loss1 += buf[2 * read_i];
                    sum_loss2 += buf[2 * read_i + 1];
                }
                __syncthreads();
            }
            if (threadIdx.y == 0) {
                buf[2 * threadIdx.x] = sum_loss1;
                buf[2 * threadIdx.x + 1] = sum_loss2;
            }
            __syncthreads();
            if (threadIdx.y != 0) {
                sum_loss1 = buf[2 * threadIdx.x];
                sum_loss2 = buf[2 * threadIdx.x + 1];
            }
        }
        // all threads now have the two sums over l
        U fH = (U)n2;
        U term1 = (U(1) / fH) * c_invvar;
        T* k_grad_input = grad_input + i1 * n2;
        if (gamma != NULL) {
            for (int l = thrx; l < n2; l += numx) {
                const U c_h = static_cast<U>(k_input[l]);
                const U c_loss = static_cast<U>(k_dout[l]);
                U f_grad_input = fH * c_loss * gamma[l];
                f_grad_input -= sum_loss1;
                f_grad_input -= (c_h - c_mean) * c_invvar * sum_loss2;
                f_grad_input *= term1;
                k_grad_input[l] = static_cast<T>(f_grad_input);
            }
        } else {
            for (int l = thrx; l < n2; l += numx) {
                const U c_h = static_cast<U>(k_input[l]);
                const U c_loss = static_cast<U>(k_dout[l]);
                U f_grad_input = fH * c_loss;
                f_grad_input -= sum_loss1;
                f_grad_input -= (c_h - c_mean) * c_invvar * sum_loss2;
                f_grad_input *= term1;
                k_grad_input[l] = static_cast<T>(f_grad_input);
            }
        }
    }
}

template <typename T, typename U, typename V>
void HostLayerNormGradient(const V* dout, const U* mean, const U* invvar, at::Tensor* input, int n1,
                           int n2, const V* gamma, const V* beta, double epsilon, T* grad_input,
                           V* grad_gamma, V* grad_beta) {
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    if (gamma != NULL && beta != NULL) {
        // compute grad_gamma(j) and grad_beta(j)
        const int part_size = 16;
        const dim3 threads2(32, 4, 1);
        const dim3 blocks2((n2 + threads2.x - 1) / threads2.x, part_size, 1);
        const int nshared2_a = 2 * sizeof(U) * threads2.y * threads2.y * (threads2.x + 1);
        const int nshared2_b = threads2.x * threads2.y * sizeof(U);
        const int nshared2 = nshared2_a > nshared2_b ? nshared2_a : nshared2_b;
        at::Tensor part_grad_gamma =
            at::empty({part_size, n2}, input->options().dtype(at::ScalarType::Float));
        at::Tensor part_grad_beta = at::empty_like(part_grad_gamma);
        cuComputePartGradGammaBeta<<<blocks2, threads2, nshared2, stream>>>(
            dout, input->DATA_PTR<T>(), n1, n2, mean, invvar, U(epsilon),
            part_grad_gamma.DATA_PTR<U>(), part_grad_beta.DATA_PTR<U>());

        const dim3 threads3(32, 8, 1);
        const dim3 blocks3((n2 + threads2.x - 1) / threads2.x, 1, 1);
        const int nshared3 = threads3.x * threads3.y * sizeof(U);
        cuComputeGradGammaBeta<<<blocks3, threads3, nshared3, stream>>>(
            part_grad_gamma.DATA_PTR<U>(), part_grad_beta.DATA_PTR<U>(), part_size, n1, n2,
            grad_gamma, grad_beta);
    }

    // compute grad_input
    const uint64_t maxGridY = at::cuda::getCurrentDeviceProperties()->maxGridSize[1];
    const dim3 blocks1(1, std::min((uint64_t)n1, maxGridY), 1);
    const dim3 threads1(32, 4, 1);
    int nshared = threads1.y > 1 ? threads1.y * threads1.x * sizeof(U) : 0;
    cuComputeGradInput<<<blocks1, threads1, nshared, stream>>>(
        dout, input->DATA_PTR<T>(), n1, n2, mean, invvar, U(epsilon), gamma, grad_input);
}

void cuda_layer_norm_gradient(at::Tensor* dout, at::Tensor* mean, at::Tensor* invvar,
                              at::Tensor* input, int n1, int n2, at::IntArrayRef normalized_shape,
                              at::Tensor* gamma, at::Tensor* beta, double epsilon,
                              at::Tensor* grad_input, at::Tensor* grad_gamma,
                              at::Tensor* grad_beta) {
    using namespace at;
    DISPATCH_FLOAT_HALF_AND_BFLOAT_INOUT_TYPES(
        input->scalar_type(), gamma->scalar_type(), "cuda_layer_norm_gradient_kernel",
        HostLayerNormGradient(dout->DATA_PTR<scalar_t_out>(), mean->DATA_PTR<float>(),
                              invvar->DATA_PTR<float>(), input, n1, n2,
                              // TMJ pass NULL argument for gamma, beta, grad_gamma and grad_beta
                              // if gamma Tensor is NULL on input.
                              gamma != NULL ? gamma->DATA_PTR<scalar_t_out>() : NULL,
                              gamma != NULL ? beta->DATA_PTR<scalar_t_out>() : NULL, epsilon,
                              grad_input->DATA_PTR<scalar_t_in>(),
                              gamma != NULL ? grad_gamma->DATA_PTR<scalar_t_out>() : NULL,
                              gamma != NULL ? grad_beta->DATA_PTR<scalar_t_out>() : NULL);)
}