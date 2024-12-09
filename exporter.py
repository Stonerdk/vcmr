#%%
import os
os.chdir("/root/dev/vcmr")

import torch
import coremltools as ct
import mobileclip
import numpy as np
from PIL import Image

from train import MLP

image_proj = MLP(input_dim=512, output_dim=256)
checkpoint = torch.load("./checkpoint_epoch_200_transformer_infonce.pth")
image_proj.load_state_dict(checkpoint['image_proj_state_dict'])
image_proj = image_proj.to("cpu").eval()

traced_model = torch.jit.trace(image_proj, torch.randn(1, 512))

mlmodel = ct.convert(
    traced_model,
    convert_to="mlprogram",
    minimum_deployment_target=ct.target.iOS16,
    inputs=[ct.TensorType(name="input", shape=(1, 512), dtype=np.float32)],
    outputs=[ct.TensorType(name="output", dtype=np.float32)]
)
mlmodel.save("MLP_transformer.mlpackage")

# %%
