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

import copy
import types
import os
import torch
from torch import nn
from torch import Tensor
from typing import Union, Optional, Tuple, Dict, List
import torch.nn.functional as F
from torch.nn.modules.utils import _pair, _triple
import datetime
import kpex._C as kernel

kpex_print_check = os.environ.get("KPEX_PRINT_CHECK") == "1"

class ModelParamChecker:
    # Dedicated class for model parameter validation: only perform general validation on all types of parameters,
    # record validation results, and return the overall validation status.
    # Does not modify any model parameters; returns the original model directly if validation fails.

    def __init__(self):
        # Initialize validation results (including overall status and detailed report)
        self.check_report: List[Dict] = []
        self.check_passed: bool = False  # Overall validation status (all tensors pass validation)

    # ---------------------- General Validation Helper Functions ----------------------
    def _check_invalid_values(self, tensor: torch.Tensor, param_full_name: str) -> Tuple[bool, str]:
        # Check tensor to ensure there are no NaN/Inf illegal values
        has_nan = torch.isnan(tensor).any().item()
        has_inf = torch.isinf(tensor).any().item()
        if has_nan and has_inf:
            return False, f"contains NaN & Inf illegal values"
        elif has_nan:
            return False, f"contains NaN illegal value"
        elif has_inf:
            return False, f"contains Inf illegal value"
        else:
            return True, "no illegal values"

    def _check_contiguous(self, tensor: torch.Tensor, param_full_name: str) -> Tuple[bool, str]:
        # Check if the tensor memory is contiguous
        is_contiguous = tensor.is_contiguous()
        return (True, "memory is contiguous") if is_contiguous else (False, "memory is not contiguous (call .contiguous() to optimize)")

    def _check_non_empty(self, tensor: torch.Tensor, param_full_name: str) -> Tuple[bool, str]:
        # Check if the tensor is empty (no valid elements)
        is_non_empty = tensor.numel() > 0
        if is_non_empty:
            return True, f"param is not empty, total elements num : {tensor.numel()}"
        else:
            return False, "param is empty, no valid elements"
    
    def _check_non_nested(self, tensor: torch.Tensor, param_full_name: str) -> Tuple[bool, str]:
        # Check if the tensor is nested (tensor containing other tensors)
        is_non_nested = (tensor.is_nested == False)
        return (True, "tensor is not nested") if is_non_nested else (False, "tensor is nested (not supported)")

    # ---------------------- Core: Perform Full Model Parameter Validation ----------------------
    def check_model(self, model: nn.Module) -> None:
        # Perform full parameter validation on the model (trainable + non-trainable + buffers)
        # Update self.check_passed (overall status) and self.check_report (detailed report) after validation
        
        # Reset validation results
        self.check_report = []
        self.check_passed = False
        
        # Switch model to eval mode (does not modify model state, only for safety)
        model.eval()
        
        # Traverse all modules: module_full_name (module path, e.g., fc1, layer1.layernorm), module (module instance)
        for module_full_name, module in model.named_modules():
            layer_class = module.__class__
            layer_class_name = layer_class.__name__
             
            # 2. Collate all tensors to be validated for this module (trainable + non-trainable + buffers)
            all_tensors = []
            # 2.1 Trainable parameters: tensor_name is local name (weight/bias)
            trainable_params = [(name, param, "trainable param") for name, param in module.named_parameters(recurse=True) if param.requires_grad]
            # 2.2 Non-trainable parameters: tensor_name is local name (weight/bias)
            non_trainable_params = [(name, param, "non-trainable param") for name, param in module.named_parameters(recurse=True) if not param.requires_grad]
            # 2.3 Model buffers: tensor_name is local name (weight/bias)
            buffers = [(name, buf, "module buffer param") for name, buf in module.named_buffers(recurse=True)]
            # Merge all tensors
            all_tensors = trainable_params + non_trainable_params + buffers
            
            param_full_names = set()
            unique_tensors = []
            # 3. Perform general validation on each tensor
            for tensor_local_name, tensor, tensor_type in all_tensors:
                # Splice the full parameter name (what the user refers to as fc1.weight, layer1.layernorm.bias)
                param_full_name = f"{module_full_name}.{tensor_local_name}" if module_full_name else tensor_local_name
                if param_full_name not in param_full_names:
                    param_full_names.add(param_full_name)
                    unique_tensors.append((tensor_local_name, tensor, tensor_type))

            for tensor_local_name, tensor, tensor_type in unique_tensors:
                param_full_name = f"{module_full_name}.{tensor_local_name}" if module_full_name else tensor_local_name
     
                # Initialize all validation results
                valid_values, value_msg = True, "no illegal values (param is empty, skipped)"
                valid_contig, contig_msg = True, "skipped (allowed empty param)"
                valid_non_empty, empty_msg = True, "skipped (allowed empty param)"
                valid_non_nested, nest_msg = True, "skipped (allowed empty param)"
                
                # Perform full validation
                if tensor is None:
                    continue
                valid_non_nested, nest_msg = self._check_non_nested(tensor, param_full_name)
                if valid_non_nested is True:
                    valid_values, value_msg = self._check_invalid_values(tensor, param_full_name)
                    valid_contig, contig_msg = self._check_contiguous(tensor, param_full_name)
                    valid_non_empty, empty_msg = self._check_non_empty(tensor, param_full_name)
                else:
                    value_msg = "skipped (due to nested tensor)"
                    contig_msg = "skipped (due to nested tensor)"
                    empty_msg = "skipped (due to nested tensor)"
                
                # Record single tensor validation result
                single_check_result = {
                    "module_full_name": module_full_name,  # Full module path (e.g., fc1, layer1.layernorm)
                    "layer_class_name": layer_class_name,
                    "tensor_local_name": tensor_local_name,  # Add: Record local parameter name (weight/bias)
                    "param_full_name": param_full_name,  # Add: Record full parameter name (e.g., fc1.weight)
                    "tensor_type": tensor_type,
                    "invalid_values": value_msg,
                    "contiguous": contig_msg,
                    "non_empty": empty_msg,
                    "nested": nest_msg,
                    "overall_status": "pass" if all([valid_values, valid_contig, valid_non_empty, valid_non_nested]) else "not pass"
                }
                self.check_report.append(single_check_result)
        
        # 4. Update overall validation status (all tensors' overall_status must be "pass" to consider overall pass)
        all_passed = all([item["overall_status"] == "pass" for item in self.check_report])
        self.check_passed = all_passed

    # ---------------------- New: Print Only Failed Validation Items ----------------------
    def print_failed_check_items(self) -> None:
        # Filter and print only failed validation items
        failed_items = [item for item in self.check_report if item["overall_status"] == "not pass"]
        
        print("=" * 80)
        print("Module Param Check Report (ONLY FAILED ITEMS)")
        print("=" * 80)
        print(f"[check result]: not pass (total failed items: {len(failed_items)})\n")
        
        for item in failed_items:
            print(f"[module full name (module path)]: {item['module_full_name']}")
            print(f"[layer class (module type)] : {item['layer_class_name']}")
            print(f"[param full name (full parameter name)] : {item['param_full_name']}")
            print(f"[tensor local name (local parameter name)] : {item['tensor_local_name']}")
            print(f"[tensor type] : {item['tensor_type']}")
            print(f"[invalid param check] : {item['invalid_values']}")
            print(f"[memory continuity check] : {item['contiguous']}")
            print(f"[non empty check] : {item['non_empty']}")
            print(f"[non nested check] : {item['nested']}")
            print(f"[single tensor state] : {item['overall_status']}")
            print("-" * 60)
        print("\n" + "=" * 80)

    # ---------------------- New: Save Validation Report to File ----------------------
    def save_check_report_to_file(self, file_prefix: str = "kpex_check_report") -> str:
        # Generate a unique file name with timestamp to avoid overwriting
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        file_name = f"{file_prefix}_{timestamp}.txt"
        
        with open(file_name, "w", encoding="utf-8") as f:
            # Write report header
            f.write("=" * 80 + "\n")
            f.write("All Module Param Check Report:\n")
            f.write("=" * 80 + "\n")
            f.write(f"[check result]: {'pass' if self.check_passed else 'not pass'}\n\n")
            
            # Write all validation items
            for item in self.check_report:
                f.write(f"[module full name (module path)]: {item['module_full_name']}\n")
                f.write(f"[layer class (module type)] : {item['layer_class_name']}\n")
                f.write(f"[param full name (full parameter name)] : {item['param_full_name']}\n")
                f.write(f"[tensor local name (local parameter name)] : {item['tensor_local_name']}\n")
                f.write(f"[tensor type] : {item['tensor_type']}\n")
                f.write(f"[invalid param check] : {item['invalid_values']}\n")
                f.write(f"[memory continuity check] : {item['contiguous']}\n")
                f.write(f"[non empty check] : {item['non_empty']}\n")
                f.write(f"[non nested check] : {item['nested']}\n")
                f.write(f"[single tensor state] : {item['overall_status']}\n")
                f.write("-" * 60 + "\n")
            f.write("\n" + "=" * 80 + "\n")
        
        # Return file name for user prompt
        return file_name

    # ---------------------- Original: Print Full Validation Report (Retained for Backup) ----------------------
    def print_check_report(self) -> None:
        # Print structured validation report
        print("=" * 80)
        print("All Module Param Check Report:")
        print("=" * 80)
        print(f"[check result]: {'pass' if self.check_passed else 'not pass'}\n")
        if kpex_print_check:
            for item in self.check_report:
                print(f"[module full name (module path)]: {item['module_full_name']}")
                print(f"[layer class (module type)] : {item['layer_class_name']}")
                print(f"[param full name (full parameter name)] : {item['param_full_name']}")
                print(f"[tensor local name (local parameter name)] : {item['tensor_local_name']}")
                print(f"[tensor type] : {item['tensor_type']}")
                print(f"[invalid param check] : {item['invalid_values']}")
                print(f"[memory continuity check] : {item['contiguous']}")
                print(f"[non empty check] : {item['non_empty']}")
                print(f"[non nested check] : {item['nested']}")
                print(f"[shape matching] : {item['shape_matching']}")
                print(f"[single tensor state] : {item['overall_status']}")
                print("-" * 60)
        print("\n" + "=" * 80)

    def is_check_passed(self) -> bool:
        # Return overall validation status: True=passed, False=not passed
        return self.check_passed

def optimize(model: nn.Module) -> nn.Module:
    # ---------------------- New: Model Non-Empty & Legitimacy Validation (Pre-validation) ----------------------
    # 1. Check if model is None (empty object)
    if model is None:
        print("Error: The input model is None (empty object), no optimization can be performed.")
        return model  # Return original None model
    
    # 2. Check if model is a valid instance of torch.nn.Module
    if not isinstance(model, nn.Module):
        print(f"Error: The input object is not a valid torch.nn.Module instance (type: {type(model).__name__}), no optimization can be performed.")
        return model
    
    # 3. Original: Check if model has no parameters (retain and optimize prompt)
    if len(list(model.parameters())) == 0:
        print("Warning: The input model has no trainable parameters (empty params), no optimization performed.")
        return model
    
    # ---------------------- Main Logic: Adjust Output Branch ----------------------
    checker = ModelParamChecker()
    checker.check_model(model)
    
    # Step 1: Judge validation result and handle output by branch
    if checker.is_check_passed():
        # Validation passed: Save validation report to file and print prompt
        check_file = checker.print_check_report()
        print(f"All checks passed successfully")
        
        op_model = copy.deepcopy(model)
        
        # Replace model operators
        def kudnn_linear(input, weight, bias = None):
            out = kernel.kudnn.kudnn_linear(input, weight, bias)
            return out

        def kudnn_conv2d(input: Tensor, weight: Tensor, bias: Optional[Tensor] = None, stride: Union[int, Tuple[int, int]] = 1, padding: Union[str, int, Tuple[int, int]] = 0, 
            dilation: Union[int, Tuple[int, int]] = 1, groups: int = 1) -> Tensor:
            return kernel.kudnn.kudnn_conv2d(
                input, weight, bias, stride, padding, dilation, groups
            )

        def kudnn_conv3d(input: Tensor, weight: Tensor, bias: Optional[Tensor] = None, stride: Union[int, Tuple[int, int, int]] = 1, padding: Union[str, int, Tuple[int, int, int]] = 0, 
            dilation: Union[int, Tuple[int, int, int]] = 1, groups: int = 1) -> Tensor:
            return kernel.kudnn.kudnn_conv3d(
                input, weight, bias, stride, padding, dilation, groups
            )

        def kudnn_linear_forward(self, input: Tensor) -> Tensor:
            out = kudnn_linear(input, self.weight, self.bias)
            return out

        def kudnn_conv2d_forward(self, input: Tensor) -> Tensor:
            if self.padding_mode != "zeros":
                return kudnn_conv2d(
                    F.pad(
                        input, self._reversed_padding_repeated_twice, mode=self.padding_mode
                    ),
                    self.weight,
                    self.bias,
                    self.stride,
                    _pair(0),
                    self.dilation,
                    self.groups,
                )
            return kudnn_conv2d(
                input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups
            )

        def kudnn_conv3d_forward(self, input:Tensor) -> Tensor:
            if self.padding_mode != "zeros":
                return kudnn_conv3d(
                    F.pad(
                        input, self._reversed_padding_repeated_twice, mode=self.padding_mode
                    ),
                    self.weight,
                    self.bias,
                    self.stride,
                    _triple(0),
                    self.dilation,
                    self.groups,
                )
            return kudnn_conv3d(
                input, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups
            )

        module_config = {
            nn.Linear: (kudnn_linear_forward, "Linear"),
            nn.Conv2d: (kudnn_conv2d_forward, "Conv2d"),
            nn.Conv3d: (kudnn_conv3d_forward, "Conv3d") # conv ops don't perform better than torch conv ops when input size is big, especially when using ResNet18, kudnn ops perform bad
        }
        
        # Traverse converted model modules: module_path (module path), module_instance (module instance, accurate recognition)
        for module_path, module_instance in op_model.named_modules():
            # Step 3: Match module types in the configuration dictionary
            for target_module_type, (kudnn_forward, module_name) in module_config.items():
                if isinstance(module_instance, target_module_type):
                    module_instance.forward = kudnn_forward.__get__(module_instance, target_module_type)
                    break  # Break inner loop after successful matching to improve efficiency
        
        F.linear = kudnn_linear
        F.conv2d = kudnn_conv2d
        F.conv3d = kudnn_conv3d
        
        return op_model
    else:
        # Validation failed: Print only failed items and return the original model
        checker.print_failed_check_items()
        print("\n===== Check failed, will return original model =====")
        return model