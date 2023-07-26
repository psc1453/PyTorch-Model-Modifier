import copy

import torch
from torch import nn

from ModelModifier.modifier.classes import NNModule


def quantize_tensor_with_original_scale(tensor_input: torch.Tensor, width: int) -> torch.Tensor:
    assert isinstance(tensor_input, torch.Tensor)
    assert isinstance(width, int)
    assert width > 0

    max_val = torch.max(tensor_input).item()
    min_val = torch.min(tensor_input).item()

    tensor_element_range = max_val - min_val

    if tensor_element_range == 0:
        return tensor_input
    else:
        steps = 2 ** width
        resolution = tensor_element_range / steps

        tensor_steps = tensor_input / resolution
        tensor_steps_int = torch.floor(tensor_steps)

        quantized_tensor = tensor_steps_int * resolution

        return quantized_tensor


def quantize_weight_by_channel_with_original_scale(tensor_input: torch.Tensor, width: int) -> torch.Tensor:
    assert isinstance(tensor_input, torch.Tensor)
    assert isinstance(width, int)
    assert width > 0

    input_shape = tensor_input.shape
    assert len(input_shape) == 4

    kernel_num = input_shape[0]

    quantized_tensor = torch.zeros_like(tensor_input)
    for kernel_index in range(kernel_num):
        quantized_tensor[kernel_index] = quantize_tensor_with_original_scale(tensor_input[kernel_index], width=width)

    return quantized_tensor


def quantize_model_parameters_with_original_scale(model_input: nn.Module, weight_width: int,
                                                  bias_width: int, by_channel=False) -> NNModule:
    model = copy.deepcopy(model_input)
    model_parameters = model.state_dict()
    for parameter_name in model_parameters:
        # TODO: Add support for layers without the name of conv and linear
        if 'weight' in parameter_name:
            if by_channel:
                model_parameters[parameter_name] = quantize_weight_by_channel_with_original_scale(
                    tensor_input=model_parameters[parameter_name], width=weight_width
                )
            else:
                model_parameters[parameter_name] = quantize_tensor_with_original_scale(
                    tensor_input=model_parameters[parameter_name], width=weight_width
                )
        elif 'bias' in parameter_name:
            model_parameters[parameter_name] = quantize_tensor_with_original_scale(
                tensor_input=model_parameters[parameter_name], width=bias_width
            )
        elif 'running_mean' in parameter_name:
            pass
        elif 'running_var' in parameter_name:
            pass
        elif 'num_batches_tracked' in parameter_name:
            pass
        else:
            raise KeyError('Unsupported state dict type found. (%s)' % parameter_name)
    model.load_state_dict(model_parameters)
    return model
