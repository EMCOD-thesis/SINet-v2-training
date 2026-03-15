import sys
import torch
from pathlib import Path

SINET_ROOT = Path(__file__).resolve().parent / "SINet-V2"
sys.path.insert(0, str(SINET_ROOT))
from lib.Network_Res2Net_GRA_NCD import Network

MODEL_WEIGHTS = "sinetv2_fine_tuned.pth"
OUTPUT_ONNX   = "sinetv2_fine_tuned.onnx"
INPUT_HEIGHT  = 352
INPUT_WIDTH   = 352
DEVICE        = "cpu"
OPSET         = 18

class S3Only(torch.nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

    def forward(self, x):
        _, _, _, s3 = self.base_model(x)
        return s3
    
base_model = Network()
checkpoint = torch.load(MODEL_WEIGHTS, map_location=DEVICE)
if "state_dict" in checkpoint:
    checkpoint = checkpoint["state_dict"]

base_model.load_state_dict(checkpoint, strict=False)
base_model.eval()
base_model.to(DEVICE)
model = S3Only(base_model)
model.eval()
model.to(DEVICE)

dummy_input = torch.randn(
    1,
    3,
    INPUT_HEIGHT,
    INPUT_WIDTH,
    device=DEVICE
)
torch.onnx.export(
    model,
    dummy_input,
    OUTPUT_ONNX,
    export_params=True,
    opset_version=OPSET,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["S_3_pred"],
    dynamic_axes=None
)
print("ONNX export complete:", OUTPUT_ONNX)