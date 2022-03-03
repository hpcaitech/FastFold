#include <torch/extension.h>

at::Tensor softmax(at::Tensor input, int rows, int cols);
at::Tensor softmax_gradient(at::Tensor d_output, at::Tensor output, int rows, int cols);

at::Tensor fused_scale_mask_softmax_forward(at::Tensor input, at::Tensor mask, int rows, int cols,
                                            float scale);
at::Tensor fused_scale_mask_softmax_backward(at::Tensor d_output, at::Tensor input, at::Tensor mask,
                                             int rows, int cols, float scale);

at::Tensor fused_scale_mask_bias_softmax_forward(at::Tensor input, at::Tensor mask, at::Tensor bias,
                                                 int rows, int cols, float scale);
at::Tensor fused_scale_mask_bias_softmax_backward(at::Tensor d_output, at::Tensor input,
                                                  at::Tensor mask, at::Tensor bias, int rows,
                                                  int cols, float scale);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &softmax, "Softmax forward (CUDA)");
    m.def("backward", &softmax_gradient, "Softmax backward (CUDA)");

    m.def("fused_scale_mask_softmax_forward", &fused_scale_mask_softmax_forward,
          "Softmax forward (CUDA)");
    m.def("fused_scale_mask_softmax_backward", &fused_scale_mask_softmax_backward,
          "Softmax forward (CUDA)");

    m.def("fused_scale_mask_bias_softmax_forward", &fused_scale_mask_bias_softmax_forward,
          "Softmax forward (CUDA)");
    m.def("fused_scale_mask_bias_softmax_backward", &fused_scale_mask_bias_softmax_backward,
          "Softmax forward (CUDA)");
}