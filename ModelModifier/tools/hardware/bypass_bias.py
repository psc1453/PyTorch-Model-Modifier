import torch
from torch import Tensor
from torch.fx import symbolic_trace

from ModelModifier.modifier.classes import NNModule, NodeInsertMapping
from ModelModifier.modifier.utils import get_insert_config, get_node_output, set_node_input
from ModelModifier.tools.hardware.io import reshape_tensor_for_hardware_pe_output
from ModelModifier.tools.quantization.utils import quantize_tensor_with_original_scale


def bypass_bias_adder(input_tensor, bias, width):
    # Reshape the batched tensor to original shape
    tensor_buffer = reshape_tensor_for_hardware_pe_output(input_tensor, width)
    # Must be 4D tensor
    input_dimension = len(tensor_buffer.shape)
    assert input_dimension == 4, 'Expect input tensor dimension: 4, but get %d' % input_dimension

    # Convert bias list to tensor
    bias_tensor = Tensor(bias)
    # Broadcast 1D tensor to 4D
    bias_tensor_broadcast = bias_tensor[None, :, None, None]

    quantized_tensor_buffer = quantize_tensor_with_original_scale(tensor_buffer, width)
    quantized_bias_broadcast = quantize_tensor_with_original_scale(bias_tensor_broadcast, width)

    # Add the bias
    output_tensor = quantized_tensor_buffer + quantized_bias_broadcast

    return output_tensor


def insert_bias_bypass(model_input: NNModule, insert_mapping: NodeInsertMapping) -> torch.fx.GraphModule:
    # Generate necessary components
    model_state_dict = model_input.state_dict()
    symbolic_traced_module = model_input
    if not isinstance(model_input, torch.fx.GraphModule):
        symbolic_traced_module = symbolic_trace(model_input)
    symbolic_traced_module_dict = dict(symbolic_traced_module.named_modules())
    symbolic_traced_module_graph = symbolic_traced_module.graph

    latest_node_is_new_inserted = False
    for current_node in symbolic_traced_module_graph.nodes:
        # Skip an iteration if the last iteration inserts a new node, because this iteration is the new node
        if latest_node_is_new_inserted:
            # Next iteration will not enter this branch, it will be the originally existed node
            latest_node_is_new_inserted = False
        # Only originally existed node can Enter this branch.
        else:
            insert_config = get_insert_config(current_node, symbolic_traced_module_dict, insert_mapping)
            # If this node match the patter, a new node needs to be inserted after it
            if insert_config.should_insert:
                # Get the next original node
                next_origin_node = current_node.next
                # Create temporary pointer for inserting
                with symbolic_traced_module_graph.inserting_after(current_node):
                    # Cache the bias value of the current node
                    bias_value = model_state_dict[current_node.target + '.bias'].data
                    # Convert bias tensor to list
                    # ONLY LIST CAN BE PASSED TO "kwargs", AND SHOULD BE GENERATED BEFORE PASSED INTO IT
                    bias_list = bias_value.tolist()
                    # Set bias of current node in state dict to 0
                    model_state_dict[current_node.target + '.bias'] = torch.zeros_like(bias_value)
                    # Load the new state dict
                    model_input.load_state_dict(model_state_dict)
                    parameters = insert_config.function_package.parameter_dict
                    parameters.update({'bias': bias_list})
                    # Create new node after current node
                    new_node = symbolic_traced_module_graph.call_function(insert_config.function_package.function,
                                                                          kwargs=parameters)
                    # Set the input of the new node to the output of the current node
                    set_node_input(new_node, get_node_output(current_node))
                    # Get the output of the new node
                    new_node_output = get_node_output(new_node)
                    # Link the output of the new node to the input of the next original node
                    set_node_input(next_origin_node, new_node_output)

    symbolic_traced_module_graph.lint()
    return torch.fx.GraphModule(model_input, symbolic_traced_module_graph)
