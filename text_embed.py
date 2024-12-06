import os
import torch
import numpy as np
from tqdm import tqdm
import clip
from typing import List, Dict
import pickle
from glob import glob

def batched(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()

    for tt in ["train", "test"]:
        output_dir = f"./music_features_{tt}"
        os.makedirs(output_dir, exist_ok=True)

        with open(f"{tt}_captions.pkl", "rb") as f:
            captions = pickle.load(f)

        output_flist = glob(os.path.join(output_dir, '*.npy'))
        processed_stems = set(os.path.splitext(os.path.basename(f))[0] for f in output_flist)
        stems, captions_list = zip(*[(stem, caption) for stem, caption in captions.items() if stem not in processed_stems])
        print(len(stems))

        batch_size = 512

        for batch_stems, batch_captions in tqdm(zip(batched(stems, batch_size), batched(captions_list, batch_size)), total=(len(stems) - 1)//batch_size + 1):
            texts = clip.tokenize(batch_captions, truncate=True).to(device)

            with torch.no_grad():
                text_features = model.encode_text(texts)
                text_features = text_features.cpu().numpy()

            for stem, feature in zip(batch_stems, text_features):
                output_path = os.path.join(output_dir, f"{stem}.npy")
                np.save(output_path, feature)
