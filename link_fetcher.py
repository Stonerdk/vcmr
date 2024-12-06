import os
import tqdm
import subprocess
import pandas as pd
import concurrent
from concurrent.futures import ThreadPoolExecutor
import sys

df = pd.read_csv("./Spotify_Youtube.csv")
acc = []

def download_video(url,f):
    try:
        result = subprocess.check_output(f"yt-dlp -f 'worst[height>=360][ext=mp4]' -g {url}", shell=True).decode().strip()
        df.loc[df["Url_youtube"] == url, "Raw_Url"] = result
        print(f"{url}, {result}", file=f)
    except subprocess.CalledProcessError as e:
        print(f"{url}, FAIL", file=f)
chunk_size = 256


def process_chunk(start_index):
    acc = []
    chunk = df.iloc[start_index * chunk_size:min(start_index * chunk_size + chunk_size, len(df))]
    f = open(f"./links/dv{start_index}.txt", "a")
    with ThreadPoolExecutor(max_workers=16) as executor:
        futures = {executor.submit(download_video, url,f): url for url in chunk["Url_youtube"].dropna()}
        for future in tqdm.tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            url = futures[future]
            try:
                future.result()
            except Exception as e:
                print(f"Error processing {url}: {e}")
    for a in acc:
        print(a)
    for a in acc:
        print(a, file=f)
    f.close()

df.to_csv("./dataset_with_raw.csv")

if __name__ == "__main__":
    for i in range(0, 81):
        process_chunk(i)
