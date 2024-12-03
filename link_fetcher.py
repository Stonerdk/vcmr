import os
import tqdm
import subprocess
import pandas as pd
import concurrent
from concurrent.futures import ThreadPoolExecutor
import sys


os.chdir("/root/dev/vcmr")
df = pd.read_csv("./Spotify_Youtube.csv")

acc = []

def download_video(url,f):
    try:
        result = subprocess.check_output(f"yt-dlp -f 'worst[height>=360][ext=mp4]' -g {url}", shell=True).decode().strip()
        print(f"{url}, {result}", file=f)
    except subprocess.CalledProcessError as e:
        print(f"{url}, FAIL", file=f)
chunk_size = 256

def process_chunk(start_index):
    f = open(f"./links/downloaded_videos_chunk{start_index}.txt", "a")
    acc = []
    chunk = df.iloc[start_index:start_index + chunk_size]
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

if __name__ == "__main__":
    argv = sys.argv
    if len(argv) < 2:
        print("Usage: python link_fetcher.py <start_index>")
        sys.exit(1)
    start_index = int(argv[1])
    process_chunk(start_index)