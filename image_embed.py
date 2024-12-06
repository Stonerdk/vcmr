import os
from glob import glob
from PIL import Image
import torch
import clip
import numpy as np
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

os.makedirs('./features', exist_ok=True)

def cl_batch(image_files):
    images = []
    for imagefile in image_files:
        image = Image.open(imagefile).convert('RGB')  # 이미지 모드 통일
        image = preprocess(image)
        images.append(image)
    images = torch.stack(images).to(device)
    with torch.no_grad():
        image_features = model.encode_image(images)
    return image_features.cpu().numpy()



batch_size = 64

for tt in ["train", "test"]:
    image_files = glob(f"./images_{tt}/*.jpg")
    for i in tqdm(range(0, len(image_files), batch_size)):
        batch_files = image_files[i:min(len(image_files), i+batch_size)]
        file_stems = [os.path.basename(f).split(".")[0] for f in batch_files]

        batch_files_to_process = []
        file_stems_to_process = []
        for f, stem in zip(batch_files, file_stems):
            feature_path = f'./image_features_{tt}/{stem}.npy'
            if not os.path.exists(feature_path):
                batch_files_to_process.append(f)
                file_stems_to_process.append(stem)

        if batch_files_to_process:
            features = cl_batch(batch_files_to_process)
            for stem, feature in zip(file_stems_to_process, features):
                feature_path = f'./image_features_{tt}/{stem}.npy'
                np.save(feature_path, feature)