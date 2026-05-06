import torch
import torch.nn as nn
import torchvision.models as models
import pandas as pd
from pathlib import Path

def generate_profile():
    print("Loading PyTorch ResNet-50...")
    # model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    
    # model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    # import vgg16 for comparison
    model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
    model.eval()

    records = []
    layer_idx = 0
    cumulative_flops = 0.0

    def hook(module, input_tensor, output_tensor, name):
        nonlocal layer_idx, cumulative_flops
        
        if not isinstance(output_tensor, torch.Tensor):
            return

        in_shape = list(input_tensor[0].shape)
        out_shape = list(output_tensor.shape)
        
        out_mb = output_tensor.nelement() * output_tensor.element_size() / (1024 * 1024)

        flops = 0
        if isinstance(module, nn.Conv2d):
            out_h, out_w = out_shape[2], out_shape[3]
            flops = out_shape[0] * module.out_channels * out_h * out_w * \
                    (module.in_channels // module.groups) * module.kernel_size[0] * module.kernel_size[1]
            
        elif isinstance(module, nn.Linear):
            flops = out_shape[0] * module.in_features * module.out_features
           
        elif isinstance(module, (nn.BatchNorm2d, nn.ReLU, nn.MaxPool2d, nn.AdaptiveAvgPool2d)):
            flops = output_tensor.nelement()  

        mflops = flops / 1e6
        cumulative_flops += mflops

        records.append({
            "layer_id": layer_idx,
            "layer_name": name,
            "module_type": type(module).__name__,
            "input_shape": str(in_shape), 
            "compute_cost_mflops": round(mflops, 4),
            "output_size_mb": round(out_mb, 6),
            "cumulative_cost_mflops": round(cumulative_flops, 4)
        })
        layer_idx += 1

    handles = []
    for name, module in model.named_modules():
        if len(list(module.children())) == 0:
            handles.append(module.register_forward_hook(
                lambda m, i, o, n=name: hook(m, i, o, n)
            ))

    print("Running real forward pass to profile layers...")
    dummy_input = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        model(dummy_input)

    for h in handles:
        h.remove()

    df = pd.DataFrame(records)
    output_path = Path(__file__).parent / "resnet50_profile.csv"
    df.to_csv(output_path, index=False)
    print(f"Profile saved to {output_path} with {len(df)} layers and {cumulative_flops:.1f} Total MFLOPs.")

if __name__ == "__main__":
    generate_profile()