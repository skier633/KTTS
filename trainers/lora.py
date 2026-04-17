import torch
import torch.nn as nn
import torch.nn.functional as F

class LoraLinear(nn.Module):
    def __init__(self, base_layer, r=8, lora_alpha=16, lora_dropout=0.05):
        super().__init__()
        self.base_layer = base_layer
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r

        # --- Handle Conv1D vs Linear differences ---
        if hasattr(base_layer, 'in_features'):
            # Standard nn.Linear
            in_features = base_layer.in_features
            out_features = base_layer.out_features
        elif hasattr(base_layer, 'nx'):
            # GPT-2 Conv1D (nx is input, nf is output)
            in_features = base_layer.nx
            out_features = base_layer.nf
        else:
            raise AttributeError(f"Unsupported layer type: {type(base_layer)}")

        # Define the Low-Rank Matrices
        self.lora_A = nn.Parameter(torch.randn(in_features, r)*0.01)
        self.lora_B = nn.Parameter(torch.zeros(r, out_features))

        self.dropout = nn.Dropout(p=lora_dropout)

        # Freeze the original weights
        for param in self.base_layer.parameters():
            param.requires_grad = False

        # Initialization
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.normal_(self.lora_B, mean=0.0, std=0.001) #version4b

    def forward(self, x):
        # 1. Base path
        res = self.base_layer(x)
    
        # 2. LoRA path
        # For Conv1D, x is [B, S, In].
        # lora_A should be [In, R], lora_B should be [R, Out]
        # This ensures the output is [B, S, Out]
        lora_out = torch.matmul(torch.matmul(x, self.lora_A), self.lora_B)

        #connection_hack = (self.lora_A.sum() * 0.0) + (self.lora_B.sum() * 0.0)

        # 3. Combine
        return res + (lora_out * self.scaling) #+ connection_hack 

class LoraLayerNorm(nn.Module):
    def __init__(self, base_norm, r=8, lora_alpha=16):
        super().__init__()
        self.base_norm = base_norm  # This is the original frozen LayerNorm
        self.hidden_size = base_norm.normalized_shape[0]
        self.scaling = lora_alpha / r

        # LoRA for the Weight (gamma)
        self.lora_A_w = nn.Parameter(torch.randn(self.hidden_size, r)*0.01)
        self.lora_B_w = nn.Parameter(torch.zeros(r, self.hidden_size))

        # LoRA for the Bias (beta)
        self.lora_A_b = nn.Parameter(torch.randn(self.hidden_size, r)*0.01)
        self.lora_B_b = nn.Parameter(torch.zeros(r, self.hidden_size))

    # This fixes the AttributeError
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base_norm, name)

    def forward(self, x):
        # 1. Original normalization
        base_output = self.base_norm(x)

        # 2. Weight Delta: (Batch, Seq, 1280) @ (1280, R) @ (R, 1280)
        # Result is (Batch, Seq, 1280) - matches base_output perfectly
        weight_delta = (x @ self.lora_A_w) @ self.lora_B_w

        # 3. Bias Delta: (1280, R) @ (R, 1280) -> (1280, 1280)
        # BUT LayerNorm bias must be a vector of (1280).
        # We sum or mean across one dimension to collapse the matrix into a vector.
        # Alternatively, we just use lora_A_b as a projection for a fixed vector.

        # The most stable way for Bias LoRA in LayerNorm:
        bias_delta = (self.lora_A_b @ self.lora_B_b).mean(dim=0) # Collapses to (1280,)

        # 4. Combine
        return base_output + (weight_delta * self.scaling) + (bias_delta * self.scaling)

def apply_lora_to_finalnorm(model, r=8, lora_alpha=16):
    for name, module in model.named_modules():
        # Target Attention (c_attn and output projection)
        if 'ln_f' in name :
            parent_name = name.rsplit('.', 1)[0]
            child_name = name.rsplit('.', 1)[1]
            parent = dict(model.named_modules())[parent_name]

            # Swap with your LoraLinear class
            target = getattr(parent, child_name)
            new_layer = LoraLayerNorm(target, r=r, lora_alpha=lora_alpha)
            setattr(parent, child_name, new_layer)
            print("apply lora to ",name)

    # Target the custom final_norm layer
    if hasattr(model, 'final_norm'):
        target = model.final_norm
        # Wrap it in the LoraLayerNorm class we discussed
        #model.final_norm = LoraLayerNorm(target, r=r, lora_alpha=lora_alpha)
        model.add_module("final_norm", LoraLayerNorm(target))

        print("LoRA successfully applied to 'final_norm'")


def apply_lora_to_gpt(model, r=8, lora_alpha=16):
    for name, module in model.named_modules():
        # Target Attention (c_attn and output projection)
        #if 'attn.c_attn' in name or 'attn.c_proj' in name:
        if 'attn.c_attn' in name:
            parent_name = name.rsplit('.', 1)[0]
            child_name = name.rsplit('.', 1)[1]
            parent = dict(model.named_modules())[parent_name]

            # Swap with your LoraLinear class
            target = getattr(parent, child_name)
            new_layer = LoraLinear(target, r=r, lora_alpha=lora_alpha)
            setattr(parent, child_name, new_layer)

        if 'attn.c_proj' in name:
            parent_name = name.rsplit('.', 1)[0]
            child_name = name.rsplit('.', 1)[1]
            parent = dict(model.named_modules())[parent_name]

            # Swap with your LoraLinear class
            target = getattr(parent, child_name)
            new_layer = LoraLinear(target, r=r, lora_alpha=lora_alpha)
            setattr(parent, child_name, new_layer)

@torch.no_grad()
def apply_qk_only_mask(model):

    hidden_dim = 1280
    for name, module in model.named_modules():
        # Target only the fused attention layers
        if "c_attn" in name and hasattr(module, 'lora_B'):
            # lora_B shape is [r, 3 * hidden_dim] 
            # or [3 * hidden_dim, r] depending on your implementation
            # We want to zero out the indices from [2*hidden_dim : 3*hidden_dim]
            
            # If your lora_B is [3840, 64]:
            module.lora_B[2*hidden_dim : 3*hidden_dim, :].fill_(0)
            #module.lora_B[0*hidden_dim : 1*hidden_dim, :].fill_(0)

            # 1-2 save the prompt speaker id

            # If your lora_B is [64, 3840]:
            # module.lora_B[:, 2*hidden_dim : 3*hidden_dim].fill_(0)
            
            print(f"✅ Muted Value (V) path for {name}")

        '''
        # Target Feed-Forward (MLP) layers
        if 'mlp.c_fc' in name or 'mlp.c_proj' in name:
            parent_name = name.rsplit('.', 1)[0]
            child_name = name.rsplit('.', 1)[1]
            parent = dict(model.named_modules())[parent_name]

            target = getattr(parent, child_name)
            new_layer = LoraLinear(target, r=r, lora_alpha=lora_alpha)
            setattr(parent, child_name, new_layer)
        '''

@torch.no_grad()
def apply_surgical_lora(model, start_layer=7, end_layer=18):
    # Iterate through the blocks in the GPT model
    # Usually model.gpt.h or model.transformer.h
    for i, block in enumerate(model.gpt.h):
        # Check if the current layer index is OUTSIDE our target range
        if i < start_layer or i > end_layer:
            # Search for LoRA layers within this block and mute them
            for name, module in block.named_modules():
                if hasattr(module, 'scaling'):
                    module.scaling = 0.0
            print(f"Muted LoRA for Layer {i}")
        else:
            # Ensure target layers are active (adjust scale as needed)
            for name, module in block.named_modules():
                if hasattr(module, 'scaling'):
                    module.scaling = 2.0
            print(f"✅ LoRA ACTIVE for Layer {i}")


def freeze_model(model):
    lora_params = []
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True
            lora_params.append(param)
        else:
            param.requires_grad = False
    return lora_params


def save_lora_adapter(model, path="accent_adapter.pth"):
    lora_state_dict = {k: v for k, v in model.state_dict().items() if "lora_" in k}
    torch.save(lora_state_dict, path)
    print(f"Saved adapter to {path}")


