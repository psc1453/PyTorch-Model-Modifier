import torch
import torch.nn as nn
from torch import Tensor

from ModelModifier.modifier.classes import FunctionPackage, NodeInsertMappingElement, NodeInsertMapping, NNModule
from ModelModifier.modifier.utils import insert_before
from ModelModifier.tools.hardware.io import reshape_tensor_for_hardware_pe_input, reshape_tensor_for_hardware_pe_output
from ModelModifier.tools.hardware.bypass_bias import insert_bias_bypass


class PSC(nn.Module):
    def __init__(self):
        super(PSC, self).__init__()
        self.layer1 = nn.Conv2d(in_channels=4, out_channels=2, kernel_size=3)

    def forward(self, x):
        return self.layer1(x)


test_tensor = Tensor([
    [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9]
    ],
    [
        [10, 11, 12],
        [13, 14, 15],
        [16, 17, 18]
    ],
    [
        [19, 20, 21],
        [22, 23, 24],
        [25, 26, 27]
    ],
    [
        [28, 29, 30],
        [31, 32, 33],
        [34, 35, 36]
    ]
]
)


def bypass_bias_adder(input_tensor, bias):
    # Reshape the batched tensor to original shape
    tensor_buffer = reshape_tensor_for_hardware_pe_output(input_tensor)
    # Must be 4D tensor
    input_dimension = len(tensor_buffer.shape)
    assert input_dimension == 4, 'Expect input tensor dimension: 4, but get %d' % input_dimension

    # Convert bias list to tensor
    bias_tensor = Tensor(bias)
    # Broadcast 1D tensor to 4D
    bias_tensor_broadcast = bias_tensor[None, :, None, None]
    # TODO: Modify to quantized version
    # Add the bias
    output_tensor = tensor_buffer + bias_tensor_broadcast
    return output_tensor


model = PSC().eval()

input_tensor_1 = test_tensor
input_tensor_2 = input_tensor_1 + 36
batch_tensor = torch.stack([input_tensor_1, input_tensor_2], dim=0)
reshaped = reshape_tensor_for_hardware_pe_input(batch_tensor)

before_mapping = NodeInsertMapping()
reshape_function_package = FunctionPackage(reshape_tensor_for_hardware_pe_input)
conv2d_config = NodeInsertMappingElement(torch.nn.Conv2d, reshape_function_package)
before_mapping.add_config(conv2d_config)

after_mapping = NodeInsertMapping()
reshape_function_package = FunctionPackage(reshape_tensor_for_hardware_pe_output)
conv2d_config = NodeInsertMappingElement(torch.nn.Conv2d, reshape_function_package)
after_mapping.add_config(conv2d_config)

bypass_mapping = NodeInsertMapping()
reshape_function_package = FunctionPackage(bypass_bias_adder)
conv2d_config = NodeInsertMappingElement(torch.nn.Conv2d, reshape_function_package)
bypass_mapping.add_config(conv2d_config)

model = insert_before(model_input=model, insert_mapping=before_mapping)
# model = insert_after(model_input=model, insert_mapping=after_mapping)
model = insert_bias_bypass(model_input=model, insert_mapping=bypass_mapping)

out = model(reshaped)
out = reshape_tensor_for_hardware_pe_output(out)
print(out)
