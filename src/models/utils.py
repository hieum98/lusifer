from typing import Optional
import torch
import torch.nn as nn
from transformers import PreTrainedModel, Conv1D


def find_all_linear_names(model: nn.Module, quantization: Optional[bool] = False):
    if not isinstance(model, PreTrainedModel):
        raise ValueError("Model must be an instance of `transformers.PreTrainedModel`")
    
    if quantization:
        from bitsandbytes.nn import Linear4bit

        cls = (Linear4bit, Conv1D)
    else:
        cls = (torch.nn.Linear, Conv1D)

    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.rsplit(".", 1)[-1]  # get the base name
            lora_module_names.add(names)
            
    if "lm_head" in lora_module_names:  
        lora_module_names.remove("lm_head")

    # ignore the last classification head for text generation models
    output_emb = model.get_output_embeddings()
    if output_emb is not None:
        last_module_name = [name for name, module in model.named_modules() if module is output_emb][0]
        lora_module_names -= {last_module_name}
        
    return list(lora_module_names)