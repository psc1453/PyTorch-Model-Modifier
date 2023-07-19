import torch

from Demo.models import FSRCNN
from ModelModifier.modifier.classes import NodeInsertMapping, NodeInsertMappingElement, FunctionPackage
from ModelModifier.modifier.utils import insert_after, insert_before
from ModelModifier.tools.quantization.utils import quantize_model_parameters_with_original_scale, \
    quantize_tensor_with_original_scale

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

    quantized_before = insert_before(model_input=quantized_by_parameters_model, insert_mapping=mapping)
    quantized_after = insert_after(model_input=quantized_by_parameters_model, insert_mapping=mapping)

    print('Original Output: \n', model(test_input))
    print('Quantized Before Conv2D Output: \n', quantized_before(test_input))
    print('Quantized After Conv2D Output: \n', quantized_after(test_input))

    print('Graph of Quantizing Before Conv2D:')
    quantized_before.print_readable()

    print('Graph of Quantizing After Conv2D:')
    quantized_after.print_readable()
