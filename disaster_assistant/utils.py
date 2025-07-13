
import warnings

import matplotlib.pyplot as plt
import cv2
from pathlib import Path
from PIL import Image
import numpy as np
import time, threading
import json

from cv2_enumerate_cameras import enumerate_cameras
from qai_hub_models.utils.asset_loaders import load_image


class QHealthConfig(object):
    """Configuration class for QHealth

    """
    # GUI input,
    # command line input
    # config file
    def __init__(self, config):
        super().__init__()
        self.config = config
        # set save_dir where trained model and log will be saved.
        output_folder = config['output_folder']
        self.convert_durations()
        for key, value in self.config.items():
            setattr(self, key, value)
        self.output_folder = Path(output_folder)
        if self.output_folder:
            self.output_folder.mkdir(parents=True, exist_ok=True)
        # make directory for saving checkpoints and log.


    def __getitem__(self, name):
        """Access items like ordinary dict."""
        return self.config[name]

    def convert_durations(self):
        new_config = self.config.copy()
        for key, value in self.config.items():
            if '_min' in key:
                new_key = key.replace('_min', '_s')
                new_config[new_key] = 60 * value
            if '_h' in key:
                new_key = key.replace('_h', '_s')
                new_config[new_key] = 3600 * value
                new_key = key.replace('_h', '_min')
                new_config[new_key] = 60 * value

        self.config = new_config

    def setup_debug_figure(self):
        self.debug_fig, self.debug_ax = plt.subplots(1)
        plt.tight_layout()

    def set_new_frame(self, frame_np):
        self.debug_ax.clear()
        self.debug_ax.set_axis_off()

        self.debug_ax.imshow(frame_np)

    def show_new_frame(self):
        self.debug_fig.canvas.flush_events()
        self.debug_fig.canvas.draw()
        self.debug_fig.show()

    def get_camera_names(self):
        for camera_info in enumerate_cameras(cv2.CAP_MSMF):
            if self.config['enable_degree0_cam'] and self.config['degree0_cam_idx'] == camera_info.index:
                print(f'Degree 0 Camera: {camera_info.name}')
                self.config['degree0_cam_name'] = camera_info.name

            if self.config['enable_degree45_cam'] and self.config['degree45_cam_idx'] == camera_info.index:
                print(f'Degree 45 Camera: {camera_info.name}')
                self.degree45_cam_name = camera_info.name

        if self.config['enable_degree45_cam'] and self.config['degree45_cam_name'] is None:
            # camera not found, disable
            self.config['enable_degree45_cam'] = False
            warnings.warn('45 degree camera not found, disabled')

        if self.config['enable_degree0_cam'] and self.config['degree0_cam_name'] is None:
            # camera not found, disable
            self.config['enable_degree0_cam'] = False
            warnings.warn('0 degree camera not found, disabled')


    @classmethod
    def from_config_file(cls, cfg_fname):
        """
        Initialize this class from some cli arguments.
        """
        cfg_fname = Path(cfg_fname)
        with cfg_fname.open('rt') as handle:
            config = json.load(handle)

        return cls(config)
        
def get_frame(config:QHealthConfig, use_camera=True, is_bad=True):
    """get one frame based on configuration

    :param config:
    :param use_camera:
    :param is_bad: a flag for labeling the image is good pose or bad pose
    :return:
    """

    if use_camera:
        print("Grabbing webcam image")

        warmup_threshold = 70

        max_try = 30 # 1 s
        if config.enable_degree0_cam:
            cap = cv2.VideoCapture(config.degree0_cam_idx, cv2.CAP_DSHOW)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)

            n_try = 10
            ret, frame = cap.read()
            while np.mean(frame)<warmup_threshold:
                # assume this is black
                ret, frame = cap.read()
                n_try += 1
                # assume 30 fps
                time.sleep(0.03)
                if n_try > max_try:
                    ret = False
                    break
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_pil_0 = Image.fromarray(frame)
            else:
                # faild to grab one frame
                frame_pil_0 = None

            cap.release()
        else:
            frame_pil_0 = None

        if config.enable_degree45_cam:
            cap = cv2.VideoCapture(config.degree45_cam_idx, cv2.CAP_DSHOW)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
           #cap.set(cv2.CAP_PROP_EXPOSURE, -3)
            # take two pictures
            # clear the blank buffers
            ret, frame = cap.read()
            n_try = 0

            while np.mean(frame)<warmup_threshold:
                # assume this is black
                ret, frame = cap.read()
                n_try += 1
                # assume 30 fps
                time.sleep(0.03)
                if n_try > max_try:
                    ret = False
                    break

            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_pil_45 = Image.fromarray(frame)
            else:
                frame_pil_45 = None
            cap.release()
        else:
            frame_pil_45 = None


    else:
        image_root = Path('./test_photos')
        if config.enable_degree0_cam:
            # grab one test photo
            if is_bad:
                image_path_0 = image_root / f'degree0'/'bad_1.jpg'
            else:
                image_path_0 = image_root / f'degree0'/'good_1.jpg'
            frame_pil_0 = load_image(image_path_0)
        else:
            frame_pil_0 = None

        if config.enable_degree45_cam:
            # grab one test photo
            if is_bad:
                image_path_45 = image_root / f'degree45'/'bad_1.jpg'
            else:
                image_path_45 = image_root / f'degree45' / 'good_1.jpg'
            frame_pil_45 = load_image(image_path_45)
        else:
            frame_pil_45 = None


    return frame_pil_0, frame_pil_45
