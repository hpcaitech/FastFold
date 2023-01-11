/******************************************************************************
###############################################################################
# Copyright (C) 2020-2021 Habana Labs, Ltd. an Intel Company
###############################################################################
*******************************************************************************/

#include "hpu_custom_op.h"
#include <torch/extension.h>
#include <perf_lib_layer_params.h>

struct SoftMaxParam
{
    int32_t axis;
    bool with_bias;
};

bool register_fusedsoftmax() {
    // Registering custom_op::fusedsoftmax
    // inputs desc
    habana::custom_op::InputDesc input_a_desc{
        habana::custom_op::input_type::TENSOR, 0};
    habana::custom_op::InputDesc input_b_desc{
        habana::custom_op::input_type::TENSOR, 1};
    habana::custom_op::InputDesc input_d_desc{
        habana::custom_op::input_type::USER_PARAMS, 2};
    std::vector<habana::custom_op::InputDesc> inputs_desc{
        input_a_desc, input_b_desc, input_d_desc};

    // output desc
    // output shape callback
    auto output_size_lambda =
        [](const at::Stack& inputs) -> std::vector<int64_t> {
      auto self = inputs[0].toTensor(); // input
      std::vector<int64_t> result_sizes = self.sizes().vec();
      return result_sizes;
    };

    habana::custom_op::OutputDesc output_desc{
        0, c10::ScalarType::Float, output_size_lambda};

    std::vector<habana::custom_op::OutputDesc> outputs_desc{
        output_desc};

    // user param callback
    auto user_params_lambda = [](const at::Stack& inputs, size_t& size) {
      HPU_PARAMS_STUB(SoftMaxParam);
      params->with_bias = false;
      int dim = inputs[2].toInt();
      if (dim > 0)
        params->axis = inputs[0].toTensor().dim() - dim - 1;
      else
        params->axis = - dim - 1;

      return params;
    };

    // actual register
    REGISTER_CUSTOM_OP_ATTRIBUTES(
        "custom_op::fusedsoftmax", //schema name
#ifdef GAUDI2
        "fusedsoftmax_fwd_f32_gaudi2", // guid
#else
	"fusedsoftmax_fwd_f32", // guid
#endif
        inputs_desc,
        outputs_desc,
        user_params_lambda);
    std::cout << "cpp registered custom_op::fusedsoftmax\n";
    return true;
}

bool register_fusedsoftmax_bias() {
    // Registering custom_op::fusedsoftmax
    // inputs desc
    habana::custom_op::InputDesc input_a_desc{
        habana::custom_op::input_type::TENSOR, 0};
    habana::custom_op::InputDesc input_b_desc{
        habana::custom_op::input_type::TENSOR, 1};
    habana::custom_op::InputDesc input_c_desc{
        habana::custom_op::input_type::TENSOR, 2};
    habana::custom_op::InputDesc input_d_desc{
        habana::custom_op::input_type::USER_PARAMS, 3};
    std::vector<habana::custom_op::InputDesc> inputs_desc{
        input_a_desc, input_b_desc, input_c_desc, input_d_desc};

    // output desc
    // output shape callback
    auto output_size_lambda =
        [](const at::Stack& inputs) -> std::vector<int64_t> {
      auto self = inputs[0].toTensor(); // input
      std::vector<int64_t> result_sizes = self.sizes().vec();
      return result_sizes;
    };

    habana::custom_op::OutputDesc output_desc{
        0, c10::ScalarType::Float, output_size_lambda};

    std::vector<habana::custom_op::OutputDesc> outputs_desc{
        output_desc};

    // user param callback
    auto user_params_lambda = [](const at::Stack& inputs, size_t& size) {
      HPU_PARAMS_STUB(SoftMaxParam);
      params->with_bias = true;
      int dim = inputs[3].toInt();
      if (dim > 0)
        params->axis = inputs[0].toTensor().dim() - dim - 1;
      else
        params->axis = - dim - 1;

      return params;
    };

    // actual register
    REGISTER_CUSTOM_OP_ATTRIBUTES(
        "custom_op::fusedsoftmax_bias", //schema name
#ifdef GAUDI2
        "fusedsoftmax_bias_fwd_f32_gaudi2", // guid
#else
        "fusedsoftmax_bias_fwd_f32", // guid
#endif
        inputs_desc,
        outputs_desc,
        user_params_lambda);
    std::cout << "cpp registered custom_op::fusedsoftmax_bias\n";
    return true;
}

bool register_custom_fusedsoftmax_backward() {
    // inputs desc
    habana::custom_op::InputDesc y_desc{
        habana::custom_op::input_type::TENSOR, 0};
    habana::custom_op::InputDesc grad_desc{
        habana::custom_op::input_type::TENSOR, 1};
    habana::custom_op::InputDesc dim_desc{
        habana::custom_op::input_type::USER_PARAMS, 2};

    std::vector<habana::custom_op::InputDesc> inputs_desc{
        y_desc, grad_desc, dim_desc};

    auto output_input_size_lambda =
        [](const at::Stack& inputs) -> std::vector<int64_t> {
      auto self = inputs[0].toTensor(); // input
      std::vector<int64_t> result_sizes = self.sizes().vec();
      return result_sizes;
    };

    habana::custom_op::OutputDesc input_grad_desc{
        0, c10::ScalarType::Float, output_input_size_lambda};

    std::vector<habana::custom_op::OutputDesc> outputs_desc{
        input_grad_desc};

    // user param callback
    auto user_params_lambda = [](const at::Stack& inputs, size_t& size) {
      HPU_PARAMS_STUB(ns_Softmax::Params);
      params->dim = 0;
      return params;
    };

    // actual register
    REGISTER_CUSTOM_OP_ATTRIBUTES(
        "custom_op::fusedsoftmax_backward", //schema name
#ifdef GAUDI2
        "softmax_bwd_f32", // guid
#else
        "softmax_bwd_f32", // guid
#endif
        inputs_desc,
        outputs_desc,
        user_params_lambda);
    std::cout << "cpp registered custom_op::fusedsoftmax_backward\n";
    return true;
}

at::Tensor fusedsoftmax_execute(
    torch::Tensor input,
    torch::Tensor mask,
    at::Scalar dim) {
  TORCH_CHECK(input.scalar_type() == c10::ScalarType::Float, "Input input_a expected to be Float tensor");
  // Registering the custom op, need to be called only once
  static bool registered = register_fusedsoftmax();
  TORCH_CHECK(registered, "fusedsoftmax kernel not registered" );
  std::vector<c10::IValue> inputs{input, mask, dim};
  // Get custom op descriptor from registry
  auto op_desc = habana::custom_op::HabanaCustomOpDescriptor::getCustomOpDescriptor("custom_op::fusedsoftmax");
  // Actual call for op execution
  std::vector<at::Tensor> output = op_desc.execute(inputs);
  // op_desc.execute will always return a vector
  return output[0];
}

at::Tensor fusedsoftmax_bias_execute(
    torch::Tensor input,
    torch::Tensor mask,
    torch::Tensor bias,
    at::Scalar dim) {
  TORCH_CHECK(input.scalar_type() == c10::ScalarType::Float, "Input input_a expected to be Float tensor");
  // Registering the custom op, need to be called only once
  static bool registered = register_fusedsoftmax_bias();
  TORCH_CHECK(registered, "fusedsoftmax_bias kernel not registered" );
  std::vector<c10::IValue> inputs{input, mask, bias, dim};
  // Get custom op descriptor from registry
  auto op_desc = habana::custom_op::HabanaCustomOpDescriptor::getCustomOpDescriptor("custom_op::fusedsoftmax_bias");
  // Actual call for op execution
  std::vector<at::Tensor> output = op_desc.execute(inputs);
  // op_desc.execute will always return a vector
  return output[0];
}

at::Tensor fusedsoftmax_backward_execute(
    torch::Tensor y,
    torch::Tensor grad,
    at::Scalar dim) {
  TORCH_CHECK(y.scalar_type() == c10::ScalarType::Float, "Input y expected to be Float tensor");
  TORCH_CHECK(grad.scalar_type() == c10::ScalarType::Float, "Input grad expected to be Float tensor");

  // Registering the custom op, need to be called only once
  static bool registered = register_custom_fusedsoftmax_backward();
  TORCH_CHECK(registered, "custom_fusedsoftmax_backward kernel not registered" );
  std::vector<c10::IValue> inputs{y, grad, dim};
  // Get custom op descriptor from registry
  auto op_desc = habana::custom_op::HabanaCustomOpDescriptor::getCustomOpDescriptor("custom_op::fusedsoftmax_backward");
  // Actual call for op execution
  std::vector<at::Tensor> output = op_desc.execute(inputs);
  // op_desc.execute will always return a vector
  return output[0];
}

TORCH_LIBRARY(custom_op, m) {
  m.def("fusedsoftmax(Tensor self, Tensor mask, Scalar dim) -> Tensor");
  m.def("fusedsoftmax_bias(Tensor self, Tensor mask, Tensor bias, Scalar dim) -> Tensor");
  m.def("fusedsoftmax_backward(Tensor y, Tensor grad, Scalar dim) -> Tensor");
}

TORCH_LIBRARY_IMPL(custom_op, HPU, m) {
  m.impl("fusedsoftmax", fusedsoftmax_execute);
  m.impl("fusedsoftmax_bias", fusedsoftmax_bias_execute);
  m.impl("fusedsoftmax_backward", fusedsoftmax_backward_execute);
}

