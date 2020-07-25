from pytube import YouTube
import os
import shutil
import math
import datetime
import glob

import cv2

from frame_extactor import FrameExtractor

australian_open = 'https://www.youtube.com/watch?v=NGu2nfQ-Itc'
french_open = 'https://www.youtube.com/watch?v=bjjJnuPReVY'
us_open = 'https://www.youtube.com/watch?v=Bc588DD6xmI'
wimbledon = 'https://www.youtube.com/watch?v=yqBOMTxMqn8'


def download_videos(list_of_youtube_links):
    list_of_new_names = ['australian_open', 'french_open', 'us_open', 'wimbledon']
    counter = 0
    for youtube_video in list_of_youtube_links:
        video = YouTube(youtube_video)
        out_file = video.streams[0].download()
        os.rename(out_file, list_of_new_names[counter] + '.mp4')
        counter = counter + 1
    return True


def create_dataset():
    videos = [fname for fname in os.listdir('.') if fname.endswith('mp4')]

    for video in videos:
        fe = FrameExtractor(video)
        fe.get_n_images(every_x_frame=100)
        fe.extract_frames(every_x_frame=100,
                  img_name=video.split(".")[0] + '_img',
                  dest_path=video.split(".")[0] + '_images')
    return True


list_of_youtube_links = [australian_open, french_open, us_open, wimbledon]

# download_videos(list_of_youtube_links=list_of_youtube_links)
create_dataset()
