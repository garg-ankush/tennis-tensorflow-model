import cv2
import datetime
import math
import os


class FrameExtractor:
    def __init__(self, video_path):
        self.video_path = video_path
        self.vid_cap = cv2.VideoCapture(video_path)
        self.n_frames = int(self.vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = int(self.vid_cap.get(cv2.CAP_PROP_FPS))

    def get_video_duration(self):
        # Get duration of video
        duration = self.n_frames/self.fps
        print(f'Duration: {datetime.timedelta(seconds=duration)}')

    def get_n_images(self, every_x_frame):
        # Number of images to extract
        n_images = math.floor(self.n_frames/every_x_frame) + 1
        print(f'Extracting every {every_x_frame} (nd/rd/th) frame would result in {n_images} images.')

    def extract_frames(self, every_x_frame, img_name, dest_path=None, img_ext='.jpg'):
        # Extract images from video
        if not self.vid_cap.isOpened():
            self.vid_cap = cv2.VideoCapture(self.video_path)
        if dest_path is None:
            dest_path = os.getcwd()
        else:
            if not os.path.isdir(dest_path):
                os.mkdir(dest_path)
                print(f'Created the following directory: {dest_path}')

        frame_cnt = 0
        img_cnt = 0

        while self.vid_cap.isOpened():

            success, image = self.vid_cap.read()

            if not success:
                break

            if frame_cnt % every_x_frame == 0:
                img_path = os.path.join(dest_path, ''.join([img_name, '_', str(img_cnt), img_ext]))
                cv2.imwrite(img_path, image)
                img_cnt += 1

            frame_cnt += 1

        self.vid_cap.release()
        cv2.destroyAllWindows()


