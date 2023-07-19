from typing import Dict

import torch
from torch.fx import symbolic_trace
from torch.fx.node import Node

from ModelModifier.modifier.classes import NodeInsertMapping, NodeInsertConfig, NNModule


def get_node_input(node: Node):
    return node.args


def set_node_input(node: Node, value):
    node.args = (value,)


def get_node_output(node: Node) -> Node:
    return node


def get_insert_config(node: Node, state_dict: Dict, node_insert_mapping: NodeInsertMapping) -> NodeInsertConfig:
    supported_operations = ['call_module']
    if node.op not in supported_operations:
        return NodeInsertConfig(should_insert=False)

    node_type = type(state_dict[node.target])

    node_mapping = node_insert_mapping.get_mapping()

    if node_type in node_mapping.keys():
        return NodeInsertConfig(should_insert=True, function_package=node_mapping[node_type])
    else:
        return NodeInsertConfig(should_insert=False)


def insert_before(model_input: NNModule, insert_mapping: NodeInsertMapping) -> torch.fx.GraphModule:
    # Generate necessary components
    symbolic_traced_module = model_input
    if not isinstance(model_input, torch.fx.GraphModule):
        symbolic_traced_module = symbolic_trace(model_input)
    symbolic_traced_module_dict = dict(symbolic_traced_module.named_modules())
    symbolic_traced_module_graph = symbolic_traced_module.graph

    for current_node in symbolic_traced_module_graph.nodes:
        insert_config = get_insert_config(current_node, symbolic_traced_module_dict, insert_mapping)
        # If this node match the patter, a new node needs to be inserted after it
        if insert_config.should_insert:
            # Get the previous original node
            previous_origin_node = current_node.prev
            # Create temporary pointer for inserting
            with symbolic_traced_module_graph.inserting_before(current_node):
                # Create new node after current node
                new_node = symbolic_traced_module_graph.call_function(insert_config.function_package.function,
                                                                      kwargs=insert_config.function_package.parameter_dict)
                # Set the input of the new node to the output of the previous original node
                set_node_input(new_node, get_node_output(previous_origin_node))
                # Get the output of the new node
                new_node_output = get_node_output(new_node)
                # Link the output of the new node to the input of the current node
                set_node_input(current_node, new_node_output)

    symbolic_traced_module_graph.lint()
    return torch.fx.GraphModule(model_input, symbolic_traced_module_graph)


def insert_after(model_input: NNModule, insert_mapping: NodeInsertMapping) -> torch.fx.GraphModule:
    # Generate necessary components
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
                    # Create new node after current node
                    new_node = symbolic_traced_module_graph.call_function(insert_config.function_package.function,
                                                                          kwargs=insert_config.function_package.parameter_dict)
                    # Set the input of the new node to the output of the current node
                    set_node_input(new_node, get_node_output(current_node))
                    # Get the output of the new node
                    new_node_output = get_node_output(new_node)
                    # Link the output of the new node to the input of the next original node
                    set_node_input(next_origin_node, new_node_output)

    symbolic_traced_module_graph.lint()
    return torch.fx.GraphModule(model_input, symbolic_traced_module_graph)


@DeprecationWarning
def insert_after_legacy(model_input: NNModule, insert_mapping: NodeInsertMapping) -> torch.fx.GraphModule:
    # Generate necessary components
    symbolic_traced_module = model_input
    if not isinstance(model_input, torch.fx.GraphModule):
        symbolic_traced_module = symbolic_trace(model_input)
    symbolic_traced_module_dict = dict(symbolic_traced_module.named_modules())
    symbolic_traced_module_graph = symbolic_traced_module.graph

    last_node = None
    last_origin_node_has_been_inserted = False
    latest_node_is_new_inserted = False
    for current_node in symbolic_traced_module_graph.nodes:
        # Skip an iteration if the last iteration inserts a new node, because this iteration is the new node
        if latest_node_is_new_inserted:
            # Next iteration will not enter this branch, it will be the originally existed node
            latest_node_is_new_inserted = False
        # Only originally existed node can Enter this branch.
        else:
            # Link the output of inserted node to current original node if last original node has been inserted
            if last_origin_node_has_been_inserted:
                set_node_input(current_node, get_node_output(last_node))
            insert_config = get_insert_config(current_node, symbolic_traced_module_dict, insert_mapping)
            # If this node match the patter, a new node needs to be inserted after it
            if insert_config.should_insert:
                # Create temporary pointer for inserting
                with symbolic_traced_module_graph.inserting_after(current_node):
                    # Create new node after current node
                    new_node = symbolic_traced_module_graph.call_function(insert_config.function_package.function,
                                                                          kwargs=insert_config.function_package.parameter_dict)
                    # Set the input of the new node to the output of the current node
                    set_node_input(new_node, get_node_output(current_node))
                    # Update pointer
                    last_node = new_node
                    # Latest node becomes the newly inserted one, and will belong to next iteration
                    # Should skip that iteration
                    latest_node_is_new_inserted = True
                    last_origin_node_has_been_inserted = True
            # Doesn't match the pattern,
            else:
                # Just update the pointer
                last_node = current_node
                # Latest node it the current one, and will belong to next iteration
                # Shouldn't skip that iteration
                latest_node_is_new_inserted = False
                # Should not to change the input of the next node
                last_origin_node_has_been_inserted = False

    symbolic_traced_module_graph.lint()
    return torch.fx.GraphModule(model_input, symbolic_traced_module_graph)
