import pickle
import random
import os
import requests
from PIL import Image
import cv2
import tqdm
from dg_util.python_utils import misc_util, video_utils, youtube_utils
from yt_dlp import YoutubeDL
import subprocess
import requests
import itertools
import traceback
import pandas as pd
import concurrent
from concurrent.futures import ThreadPoolExecutor
os.chdir("/root/dev/vcmr")

import yt_dlp
import subprocess
import os
import concurrent.futures
from urllib.parse import urlparse, parse_qs
from tqdm import tqdm

df = pd.read_csv("test_df.csv")
url_list = list(df["Url_youtube"])

# 1. 디렉토리 생성
os.makedirs('./images_test', exist_ok=True)
os.makedirs('./music_test', exist_ok=True)

def process_url(url):
    # 비디오 ID 추출
    parsed_url = urlparse(url)
    video_id = ''
    if parsed_url.hostname in ('youtu.be', 'www.youtu.be'):
        video_id = parsed_url.path[1:]
    elif parsed_url.hostname in ('www.youtube.com', 'youtube.com'):
        if parsed_url.path == '/watch':
            video_id = parse_qs(parsed_url.query).get('v', [''])[0]
        elif parsed_url.path.startswith('/embed/'):
            video_id = parsed_url.path.split('/')[2]
        elif parsed_url.path.startswith('/v/'):
            video_id = parsed_url.path.split('/')[2]
    else:
        print(f"잘못된 YouTube URL: {url}")
        return

    # 2. 세그먼트 정의
    segments = [
        {
            'start': '00:00:30',
            'duration': '10',
            'segment_num': 1,
            'frame_time': '00:00:35',
        },
        {
            'start': '00:01:20',
            'duration': '10',
            'segment_num': 2,
            'frame_time': '00:01:25',
        },
        {
            'start': '00:02:10',
            'duration': '10',
            'segment_num': 3,
            'frame_time': '00:02:15',
        },
    ]

    try:
        # 3. 스트리밍 URL 가져오기
        ydl_opts = {
            'format': 'bestvideo[height<=360]+bestaudio/best[height<=360]/worst',
            'quiet': True,
            'noplaylist': True,
            'skip_download': True,
        }

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info_dict = ydl.extract_info(url, download=False)
            if 'requested_formats' in info_dict:
                video_url = info_dict['requested_formats'][0]['url']
                audio_url = info_dict['requested_formats'][1]['url']
            else:
                video_url = info_dict['url']
                audio_url = info_dict['url']

        # 4. 각 세그먼트 처리
        for segment in segments:
            segment_num = segment['segment_num']
            audio_output = f"./music_test/{video_id}_{segment_num}.mp3"
            image_output = f"./images_test/{video_id}_{segment_num}.jpg"

            # 기존 파일 삭제
            if os.path.exists(audio_output):
                os.remove(audio_output)
            if os.path.exists(image_output):
                os.remove(image_output)

            # 오디오 추출
            audio_cmd = [
                'ffmpeg',
                '-y',
                '-ss', segment['start'],
                '-t', segment['duration'],
                '-i', audio_url,
                '-vn',
                '-acodec', 'libmp3lame',
                '-ar', '44100',
                '-ac', '2',
                audio_output,
            ]

            audio_result = subprocess.run(
                audio_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            if audio_result.returncode != 0:
                print(f"오디오 추출 실패 [{url}] 세그먼트 {segment_num}:\n{audio_result.stderr}")
                if os.path.exists(audio_output):
                    os.remove(audio_output)
                continue  # 다음 세그먼트로 이동

            # 이미지 추출
            image_cmd = [
                'ffmpeg',
                '-y',
                '-ss', segment['frame_time'],
                '-i', video_url,
                '-frames:v', '1',
                image_output,
            ]

            image_result = subprocess.run(
                image_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )

            if image_result.returncode != 0:
                print(f"이미지 추출 실패 [{url}] 세그먼트 {segment_num}:\n{image_result.stderr}")
                if os.path.exists(image_output):
                    os.remove(image_output)
                if os.path.exists(audio_output):
                    os.remove(audio_output)
                continue

            # 쌍이 모두 생성되었는지 확인
            if not (os.path.exists(audio_output) and os.path.exists(image_output)):
                if os.path.exists(audio_output):
                    os.remove(audio_output)
                if os.path.exists(image_output):
                    os.remove(image_output)
                continue

    except Exception as e:
        print(f"URL 처리 중 오류 발생 [{url}]: {e}")
        return

# 5. URL 리스트 읽기
urls = url_list

# 6. 병렬 처리 설정
max_workers = 8  # 시스템 성능에 따라 조절
with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = [executor.submit(process_url, url) for url in urls]
    for _ in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Processing URLs"):
        pass  # 결과를 확인하거나 로그를 남기고 싶다면 여기서 처리
