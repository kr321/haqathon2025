import warnings
import time
import numpy as np

from qai_hub_models.models.yolov8_det.model import YoloV8Detector

from qai_hub_models.utils.image_processing import pil_resize_pad, pil_undo_resize_pad

from yolo import load_yolov8_app, yolo_eval_and_list
from utils import get_frame, QHealthConfig

import qai_hub as hub
target_device = hub.Device("Snapdragon X Elite CRD")

# Example usage
if __name__ == '__main__':
    is_debug = True
    # config
    config_file = './config.json'
    qhealth_config = QHealthConfig.from_config_file(config_file)

    if is_debug:
        qhealth_config.setup_debug_figure()
    # Load YOLO
    yolov8_app = load_yolov8_app()

    frame_pil, _ = get_frame(qhealth_config, use_camera=True)
    yolov8_h, yolov8_w = YoloV8Detector.get_input_spec()["image"][0][2:]

    yolo_eval_and_list(yolov8_app, frame_pil, (yolov8_h, yolov8_w))