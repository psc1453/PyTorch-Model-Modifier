import torch

from Demo.models import FSRCNN
from ModelModifier.modifier.classes import NodeInsertMapping, NodeInsertMappingElement, FunctionPackage
from ModelModifier.modifier.utils import generate_quantized_module
from ModelModifier.tools.quantization import quantize_model_parameters_with_original_scale, quantize_tensor_with_original_scale

CKPT_PATH = 'fsrcnn_x3.pth'
SCALE = 3

if __name__ == '__main__':
    test_input = torch.randn([1, 1, 100, 100])

    model = FSRCNN(scale_factor=SCALE)
    quantized_by_parameters_model = quantize_model_parameters_with_original_scale(model_input=model, weight_width=8,
                                                                                  bias_width=18)
    mapping = NodeInsertMapping()
    quantize_8bit_function_package = FunctionPackage(quantize_tensor_with_original_scale, {'width': 8})
    conv2d_config = NodeInsertMappingElement(torch.nn.Conv2d, quantize_8bit_function_package)
    mapping.add_config(conv2d_config)

    new = generate_quantized_module(model_input=quantized_by_parameters_model, insert_mapping=mapping)

    print(model(test_input))
    print(new(test_input))
    new.print_readable()
