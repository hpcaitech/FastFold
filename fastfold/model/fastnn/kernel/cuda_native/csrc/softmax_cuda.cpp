#include <torch/extension.h>

at::Tensor softmax(at::Tensor input, long long rows, long long cols);
at::Tensor softmax_gradient(at::Tensor d_output, at::Tensor output, long long rows, long long cols);

at::Tensor fused_mask_softmax_forward(at::Tensor input, at::Tensor mask, long long rows,
                                      long long cols);
at::Tensor fused_mask_softmax_backward(at::Tensor d_output, at::Tensor input, at::Tensor mask,
                                       long long rows, long long cols);

at::Tensor fused_mask_bias_softmax_forward(at::Tensor input, at::Tensor mask, at::Tensor bias,
                                           long long rows, long long cols);
at::Tensor fused_mask_bias_softmax_backward(at::Tensor d_output, at::Tensor input, at::Tensor mask,
                                            at::Tensor bias, long long rows, long long cols);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &softmax, "Softmax forward (CUDA)");
    m.def("backward", &softmax_gradient, "Softmax backward (CUDA)");

    m.def("fused_mask_softmax_forward", &fused_mask_softmax_forward, "Softmax forward (CUDA)");
    m.def("fused_mask_softmax_backward", &fused_mask_softmax_backward, "Softmax forward (CUDA)");

    m.def("fused_mask_bias_softmax_forward", &fused_mask_bias_softmax_forward,
          "Softmax forward (CUDA)");
    m.def("fused_mask_bias_softmax_backward", &fused_mask_bias_softmax_backward,
          "Softmax forward (CUDA)");
}