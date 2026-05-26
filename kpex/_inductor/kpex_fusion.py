#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# KPEX is licensed under Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#        http://license.coscl.org.cn/MulanPSL2
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
# See the Mulan PSL v2 for more details.
#

import torch
from torch._inductor.pattern_matcher import PatternMatcherPass, register_lowering_pattern, CallFunction, Arg, Match
from torch._inductor.lowering import make_fallback
import torch.nn.functional as F

# Experimental feature to implement "horizontal" fusion. Written for QKV weight fusion
def fuse_linear_pass(gm: torch.fx.GraphModule, example_inputs):
    # print("Graph Placeholders:", [n.name for n in gm.graph.nodes if n.op == "placeholder"])
    # print("Example Inputs Count:", len(example_inputs))

    graph = gm.graph

    # Identify Weight Placeholders
    # In a functional graph, weights are placeholders. 
    # We need to find which placeholder belongs to which Linear.
    placeholders = [n for n in graph.nodes if n.op == "placeholder"]
    # Map the target name (L_self_...) to the actual tensor in example_inputs
    # example_inputs is a list [x, weight_q, weight_k, weight_v, ...]
    input_data_map = {}
    for i, p in enumerate(placeholders):
        # In modern PyTorch, the actual tensor is usually at the same index
        # as the placeholder in the example_inputs list.
        # We use a fallback to ensure we don't crash on metadata mismatches.
        if i < len(example_inputs):
            input_data_map[p] = example_inputs[i]

    def get_tensor_from_node(node):
        """Helper to get actual tensor data from a node."""
        if node.op == "placeholder":
            return input_data_map.get(node)
        elif node.op == "get_attr":
            # Retrieve from the module hierarchy
            return dict(gm.named_parameters())[node.target] if node.target in dict(gm.named_parameters()) else getattr(gm, node.target)
        return None

    # Match high-level functional linear calls [F.linear(x, w, b)]
    groups = {}
    for node in graph.nodes:
        if node.op == "call_function" and node.target == F.linear:
            x_input = node.args[0]
            weight_node = node.args[1]

            weight_data = get_tensor_from_node(weight_node)
            if weight_data is not None:
                groups.setdefault(x_input, []).append(node)

    fusion_count = 0

    for x_input, mm_nodes in groups.items():
        if len(mm_nodes) < 2:
            continue
        
        # print(f"--- Fusing {len(mm_nodes)} Linears for input {x_input} ---")

        # 1. Identify the original weight nodes (placeholders/attributes)
        weight_nodes = [n.args[1] for n in mm_nodes]

        # --- TOPOLOGICAL FIX: Sequential Insertion ---
        # Find the topological "latest" weight node to ensure all are defined
        node_to_idx = {node: i for i, node in enumerate(graph.nodes)}
        last_weight_node = max(weight_nodes, key=lambda n: node_to_idx[n])
        # ------------------------

        with graph.inserting_after(last_weight_node):
            # Insert a CAT operation inside the FX graph after the last weight is available
            # This bypasses all metadata, storage, and grapharg errors
            # F.linear weights are [Out, In], so we cat on dim 0
            fused_w_node = graph.call_function(
                torch.ops.aten.cat.default, 
                args=(weight_nodes, 0)
            )
            # Carry over metadata so Inductor knows these are constants
            # This triggers the "Freezing" optimization
            fused_w_node.meta['is_constant_weight'] = True
            
        # Insert Transpose specifically after the CAT node
        with graph.inserting_after(fused_w_node):
            # Second: Transpose (automatically inserted AFTER fused_w_node)
            # Since we are inside the same context, this stays below fused_w_node
            w_t = graph.call_function(
                torch.ops.aten.transpose.int, 
                args=(fused_w_node, -1, -2)
            )
            w_t.meta['is_constant_weight'] = True
       
        fusion_count += 1

        # Insert the Fused Matmul where the original Linears were
        with graph.inserting_after(mm_nodes[-1]):
            # This replaces the logic of F.linear(x, w)
            # Math: [B, S, In] @ [In, Out]
            fused_node = graph.call_function(
                torch.ops.aten.matmul.default, 
                args=(x_input, w_t)
            )

            # Slice back for Q, K, V
            curr_anchor = fused_node
            offset = 0
            for i, original_node in enumerate(mm_nodes):
                w_tensor = get_tensor_from_node(weight_nodes[i])
                width = w_tensor.shape[0]
                
                with graph.inserting_after(curr_anchor):
                    slc = graph.call_function(
                        torch.ops.aten.slice.Tensor,
                        args=(fused_node, -1, offset, offset + width)
                    )
                    original_node.replace_all_uses_with(slc)
                    # Update anchor to keep slices in order
                    curr_anchor = slc
                offset += width

        # Cleanup
        for n in mm_nodes:
            graph.erase_node(n)

    return gm

# Register extra fusion graph passes
def kpex_fusion_passes(gm, example_inputs):

    # fuse_linear_pass(gm, example_inputs) # Experimental QKV fusion

    # Create the pass container
    graph_pass = PatternMatcherPass()

    # We look for linear functional
    linear_decomposed_pattern = CallFunction(
        F.linear, 
        Arg(), # Bias
        Arg(), # Input
        Arg()  # Weight
    )
    # Now wrap that in SiLU
    silu_pattern = CallFunction(
        F.silu, 
        linear_decomposed_pattern
    )

    # Supports SILU (=SWISH) activation function. Will be generalized to any activation function supported by KuDNN
    @register_lowering_pattern(silu_pattern, pass_dict=graph_pass)
    def fused_linear_silu(match, x, weight, bias):
        # This emits the fused op directly into the LIR
        return torch.ops.kpex.linear_act(x, weight, bias)

    make_fallback(torch.ops.kpex.linear_act) # Do not decompose into primitive ops
    
    # Modify PyTorch FX GraphModule (gm) by custom transformation. 
    graph_pass.apply(gm.graph) # Apply transformation
    gm.graph.lint() # Validate graph integrity
    gm.recompile() # Regenerate executable code

    return gm