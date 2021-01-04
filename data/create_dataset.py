from pytube import YouTube
import os
import shutil
import math
import datetime
import glob

import cv2

from frame_extactor import FrameExtractor

# Video links from youtube
australian_open = 'https://www.youtube.com/watch?v=NGu2nfQ-Itc'
french_open = 'https://www.youtube.com/watch?v=bjjJnuPReVY'
us_open = 'https://www.youtube.com/watch?v=Bc588DD6xmI'
wimbledon = 'https://www.youtube.com/watch?v=yqBOMTxMqn8'


def download_videos(youtube_links):
    # Download youtube videos
    list_of_new_names = ['australian_open', 'french_open', 'us_open', 'wimbledon']
    counter = 0
    for youtube_video in youtube_links:
        video = YouTube(youtube_video)
        out_file = video.streams[0].download()

        # Rename videos
        os.rename(out_file, list_of_new_names[counter] + '.mp4')
        counter = counter + 1
    return True


def create_datasets():
    # Using frame extractor create images
    videos = [fname for fname in os.listdir('.') if fname.endswith('mp4')]

    for video in videos:
        fe = FrameExtractor(video)
        # Get images every 100 frame of the video
        fe.get_n_images(every_x_frame=100)
        # Extract images and place them in a separate folder
        fe.extract_frames(every_x_frame=100,
                          img_name=video.split(".")[0] + '_img',
                          dest_path=video.split(".")[0] + '_images')


# Youtube links
list_of_youtube_links = [australian_open, french_open, us_open, wimbledon]

# Run download videos
download_videos(youtube_links=list_of_youtube_links)

# Create datasets
create_datasets()
