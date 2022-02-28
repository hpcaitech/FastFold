#include <iostream>

#include "ATen/ATen.h"
#include "ATen/cuda/CUDAContext.h"
#include "compat.h"
#include "softmax.cuh"

#define CHECK_CUDA(x) TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) \
    CHECK_CUDA(x);     \
    CHECK_CONTIGUOUS(x)

at::Tensor softmax(at::Tensor input, int rows, int cols) {
    CHECK_INPUT(input);
    at::Tensor output = at::empty_like(input);
    fastfold::softmax::DirectLoad<at::BFloat16, float> load((at::BFloat16 *)input.data_ptr(),
                                                            int64_t(cols));
    fastfold::softmax::DirectStore<float, at::BFloat16> store((at::BFloat16 *)output.data_ptr(),
                                                              int64_t(cols));

    auto cuda_stream = at::cuda::getCurrentCUDAStream().stream();
    fastfold::softmax::DispatchSoftmax<decltype(load), decltype(store), float>(cuda_stream, load,
                                                                               store, rows, cols);

    return output;
}

at::Tensor softmax_gradient(at::Tensor d_output, at::Tensor input, int rows, int cols) {
    CHECK_INPUT(input);
    at::Tensor grad_input = at::empty_like(input);
    fastfold::softmax::DirectLoad<at::BFloat16, float> load_d((at::BFloat16 *)d_output.data_ptr(),
                                                              int64_t(cols));
    fastfold::softmax::DirectLoad<at::BFloat16, float> load((at::BFloat16 *)input.data_ptr(),
                                                            int64_t(cols));
    fastfold::softmax::DirectStore<float, at::BFloat16> store((at::BFloat16 *)grad_input.data_ptr(),
                                                              int64_t(cols));

    auto cuda_stream = at::cuda::getCurrentCUDAStream().stream();
    fastfold::softmax::DispatchSoftmaxGrad<decltype(load), decltype(load_d), decltype(store),
                                           float>(cuda_stream, load, load_d, store, rows, cols);

    return grad_input;
}

at::Tensor fused_scale_mask_softmax_forward(at::Tensor input, at::Tensor mask, int rows, int cols,
                                            float scale) {
    CHECK_INPUT(input);
    CHECK_INPUT(mask);
    int head = input.sizes()[2];
    at::Tensor output = at::empty_like(input);
    // (const SRC* src, const int8_t* mask, int64_t row_size, SRC scale)
    fastfold::softmax::ScaleMaskLoad<at::BFloat16, float> load((at::BFloat16 *)input.data_ptr(),
                                                               (at::BFloat16 *)mask.data_ptr(),
                                                               int64_t(cols), int64_t(head), scale);
    fastfold::softmax::DirectStore<float, at::BFloat16> store((at::BFloat16 *)output.data_ptr(),
                                                              int64_t(cols));

    auto cuda_stream = at::cuda::getCurrentCUDAStream().stream();
    fastfold::softmax::DispatchSoftmax<decltype(load), decltype(store), float>(cuda_stream, load,
                                                                               store, rows, cols);

    return output;
}

at::Tensor fused_scale_mask_softmax_backward(at::Tensor d_output, at::Tensor input, at::Tensor mask,
                                             int rows, int cols, float scale) {
    CHECK_INPUT(input);
    CHECK_INPUT(mask);
    int head = input.sizes()[2];
    at::Tensor grad_input = at::empty_like(input);
    fastfold::softmax::DirectLoad<at::BFloat16, float> load_d((at::BFloat16 *)d_output.data_ptr(),
                                                              int64_t(cols));
    fastfold::softmax::DirectLoad<at::BFloat16, float> load((at::BFloat16 *)input.data_ptr(),
                                                            int64_t(cols));
    // (DST* dst, const int8_t* mask, int64_t row_size, DST scale)
    fastfold::softmax::ScaleMaskStore<float, at::BFloat16> store(
        (at::BFloat16 *)grad_input.data_ptr(), (at::BFloat16 *)mask.data_ptr(), int64_t(cols),
        int64_t(head), scale);

    auto cuda_stream = at::cuda::getCurrentCUDAStream().stream();
    fastfold::softmax::DispatchSoftmaxGrad<decltype(load), decltype(load_d), decltype(store),
                                           float>(cuda_stream, load, load_d, store, rows, cols);

    return grad_input;
}

at::Tensor fused_scale_mask_bias_softmax_forward(at::Tensor input, at::Tensor mask, at::Tensor bias,
                                                 int rows, int cols, float scale) {
    CHECK_INPUT(input);
    CHECK_INPUT(mask);
    CHECK_INPUT(bias);
    int head = input.sizes()[2];
    at::Tensor output = at::empty_like(input);
    // (const SRC* src, const int8_t* mask, int64_t row_size, SRC scale)
    fastfold::softmax::ScaleMaskBiasLoad<at::BFloat16, float> load(
        (at::BFloat16 *)input.data_ptr(), (at::BFloat16 *)mask.data_ptr(),
        (at::BFloat16 *)bias.data_ptr(), int64_t(cols), int64_t(head), scale);
    fastfold::softmax::DirectStore<float, at::BFloat16> store((at::BFloat16 *)output.data_ptr(),
                                                              int64_t(cols));

    auto cuda_stream = at::cuda::getCurrentCUDAStream().stream();
    fastfold::softmax::DispatchSoftmax<decltype(load), decltype(store), float>(cuda_stream, load,
                                                                               store, rows, cols);

    return output;
}

at::Tensor fused_scale_mask_bias_softmax_backward(at::Tensor d_output, at::Tensor input,
                                                  at::Tensor mask, at::Tensor bias, int rows,
                                                  int cols, float scale) {
    CHECK_INPUT(input);
    CHECK_INPUT(mask);
    int head = input.sizes()[2];
    // CHECK_INPUT(bias);
    at::Tensor grad_input = at::empty_like(input);
    fastfold::softmax::DirectLoad<at::BFloat16, float> load_d((at::BFloat16 *)d_output.data_ptr(),
                                                              int64_t(cols));
    fastfold::softmax::DirectLoad<at::BFloat16, float> load((at::BFloat16 *)input.data_ptr(),
                                                            int64_t(cols));
    // (DST* dst, const int8_t* mask, int64_t row_size, DST scale)
    fastfold::softmax::ScaleMaskStore<float, at::BFloat16> store(
        (at::BFloat16 *)grad_input.data_ptr(), (at::BFloat16 *)mask.data_ptr(), int64_t(cols),
        int64_t(head), scale);

    auto cuda_stream = at::cuda::getCurrentCUDAStream().stream();
    fastfold::softmax::DispatchSoftmaxGrad<decltype(load), decltype(load_d), decltype(store),
                                           float>(cuda_stream, load, load_d, store, rows, cols);

    return grad_input;
}