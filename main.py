import torch
import torch.nn as nn
from torch import Tensor

from ModelModifier.tools.quantization.utils import quantize_model_parameters_with_original_scale, \
    quantize_tensor_with_original_scale
from ModelModifier.modifier.classes import FunctionPackage, NodeInsertMappingElement, NodeInsertMapping, NNModule
from ModelModifier.modifier.utils import insert_before
from ModelModifier.tools.hardware.io import reshape_tensor_for_hardware_pe_input, reshape_tensor_for_hardware_pe_output
from ModelModifier.tools.hardware.bypass_bias import insert_bias_bypass, bypass_bias_adder

from Demo.models import FSRCNN

CKPT_PATH = 'Demo/fsrcnn_x3.pth'
SCALE = 3

model = FSRCNN(scale_factor=SCALE)
model.load_state_dict(torch.load(CKPT_PATH))

quantized_by_parameters_model = quantize_model_parameters_with_original_scale(model_input=model, weight_width=8,
                                                                                  bias_width=18)
test_input = torch.randn([1, 1, 100, 100])

before_mapping = NodeInsertMapping()
reshape_function_package = FunctionPackage(reshape_tensor_for_hardware_pe_input)
conv2d_config = NodeInsertMappingElement(torch.nn.Conv2d, reshape_function_package)
before_mapping.add_config(conv2d_config)

bypass_mapping = NodeInsertMapping()
reshape_function_package = FunctionPackage(bypass_bias_adder, {'width': 18})
conv2d_config = NodeInsertMappingElement(torch.nn.Conv2d, reshape_function_package)
bypass_mapping.add_config(conv2d_config)

model = insert_before(model_input=quantized_by_parameters_model, insert_mapping=before_mapping)
model = insert_bias_bypass(model_input=model, insert_mapping=bypass_mapping)

out = model(test_input)

model.print_readable()
