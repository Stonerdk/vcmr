import os
from glob import glob
import torch.nn as nn
from PIL import Image
import torch
import clip
from tqdm import tqdm
import sys
from argparse import Namespace
import pickle

sys.path.append("/root/dev/vcmr/lp-music-caps/lpmc/music_captioning")
os.chdir("/root/dev/vcmr/lp-music-caps/lpmc/music_captioning")
from captioning import captioning, evaluate, evaluate_batch

args = Namespace(
    gpu=0,
    framework="pretrain",
    caption_type="lp_music_caps",
    max_length=128,
    num_beams=5,
    model_type="pretrain",
    audio_path=""
)

model = captioning(args)
print(torch.cuda.device_count())
model = model.cuda()

captions = {}

batch_size = 220
for tt in ["train", "test"]:
    pickle_path = f"/root/dev/vcmr/{tt}_captions.pkl"
    if os.path.exists(pickle_path):
        with open(pickle_path, "rb") as f:
            captions = pickle.load(f)
    else:
        captions = {}

    flist = list(glob(f"/root/dev/vcmr/music_{tt}/*.mp3"))
    processed_stems = set(captions.keys())
    flist = [fname for fname in flist if os.path.basename(fname).split(".")[0] not in processed_stems]

    for i in tqdm(range(0, len(flist), batch_size)):
        view = flist[i:min(i+batch_size, len(flist))]
        stems = [os.path.basename(fname).split(".")[0] for fname in view]
        inference = evaluate_batch(model, args, view)
        for stem, caption in inference.items():
            captions[stems[stem]] = caption["text"]
        with open(pickle_path, "wb") as f:
                pickle.dump(captions, f)
# for stem, caption in captions.items():
#     print(stem, caption)
print(captions)