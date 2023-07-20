import torch
from torch import Tensor

from ModelModifier.tools.quantization.utils import quantize_tensor_with_original_scale

def reshape_tensor_for_hardware_pe_input(input_tensor: Tensor, pe_num: int = 4):
    input_dimension = len(input_tensor.shape)
    assert input_dimension == 4, 'Expect input tensor dimension: 4, but get %d' % input_dimension

    batch, channel, height, width = input_tensor.shape
    expanded_channel = channel
    if channel % pe_num != 0:
        expanded_channel = ((channel // pe_num) + 1) * pe_num

    tensor_buffer = torch.zeros([batch, expanded_channel, height, width])
    tensor_buffer[:, 0: channel, :, :] = input_tensor
    output_tensor = torch.zeros(batch * pe_num, channel, height, width)

    for current_batch in range(batch):
        for current_pe in range(pe_num):
            target_batch = current_batch * pe_num + current_pe
            target_non_zero_channel_num = (channel + pe_num - 1) // pe_num
            target_channel_start = current_pe * target_non_zero_channel_num
            target_channel_end = min((current_pe + 1) * target_non_zero_channel_num, channel)

            output_tensor[target_batch, target_channel_start: target_channel_end, :, :] \
                = tensor_buffer[current_batch, target_channel_start: target_channel_end, :, :]
    return output_tensor


def reshape_tensor_for_hardware_pe_output(input_tensor: Tensor, width_accum, pe_num: int = 4):
    input_dimension = len(input_tensor.shape)
    assert input_dimension == 4, 'Expect input tensor dimension: 4, but get %d' % input_dimension

    batch, channel, height, width = input_tensor.shape
    assert batch % pe_num == 0, 'Input batch size %d is not multiples of PE number %d' % (batch, pe_num)

    output_batch = int(batch / pe_num)
    output_tensor = torch.zeros(output_batch, channel, height, width)

    quantized_input_tensor = quantize_tensor_with_original_scale(input_tensor, width_accum)

    for i in range(output_batch):
        batch_start = i * pe_num
        batch_end = (i + 1) * pe_num

        # TODO: Modify to hardware quantization sum
        output_tensor[i, :, :, :] = torch.sum(quantized_input_tensor[batch_start: batch_end, :, :, :], dim=0)

    quantized_output_tensor = quantize_tensor_with_original_scale(output_tensor, width_accum)
    return quantized_output_tensor
