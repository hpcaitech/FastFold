import importlib

fastfold_softmax_cuda = importlib.import_module("fastfold_softmax_cuda")


def softmax_cuda_kernel_wrapper(input_, mask_, bias_, rows, cols):
    if bias_ is not None:
        output = fastfold_softmax_cuda.fused_mask_bias_softmax_forward(input_, mask_, bias_, rows, cols)
    elif mask_ is not None:
        output = fastfold_softmax_cuda.fused_mask_softmax_forward(input_, mask_, rows, cols)
    else:
        output = fastfold_softmax_cuda.forward(input_, rows, cols)

    return output


def softmax_grad_cuda_kernel_wrapper(grad_output, output, mask_, rows, cols):
    if mask_ is not None:
        grad_input = fastfold_softmax_cuda.fused_mask_softmax_backward(grad_output, output, mask_, rows, cols)
    else:
        grad_input = fastfold_softmax_cuda.backward(grad_output, output, rows, cols)
    return grad_input
