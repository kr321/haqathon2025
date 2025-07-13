import time
import numpy as np
from utils import QHealthConfig, get_frame

from qai_hub_models.models.yolov8_det.app import YoloV8DetectionApp
from qai_hub_models.models.yolov8_det.model import YoloV8Detector
from qai_hub_models.models.yolov8_det.model import MODEL_ID as yolov8_model_id

from qai_hub_models.utils.image_processing import pil_resize_pad
from qai_hub_models.utils.args import (
    demo_model_from_cli_args,
    get_model_cli_parser,
    get_on_device_demo_parser,
    validate_on_device_demo_args,
)



# Class names for YOLOv8
yolo_names = {
    0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus',
    6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant',
    11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat',
    16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant',
    21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella',
    26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis',
    31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove',
    36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass',
    41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl',
    46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli',
    51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake',
    56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table',
    61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote',
    66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster',
    71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase',
    76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
}

# Load YOLOv8 model
def load_yolov8_app():
    parser = get_model_cli_parser(YoloV8Detector)
    parser = get_on_device_demo_parser(parser, add_output_dir=False)

    # 🛠️ Add required arguments
    parser.add_argument("--score-threshold", type=float, default=0.45)
    parser.add_argument("--iou-threshold", type=float, default=0.7)
    parser.add_argument("--include-postprocessing", action="store_true")  # Optional, depending on model usage

    args = parser.parse_args()
    validate_on_device_demo_args(args, yolov8_model_id)

    model = demo_model_from_cli_args(YoloV8Detector, yolov8_model_id, args)
    return YoloV8DetectionApp(model, args.score_threshold, args.iou_threshold, args.include_postprocessing)
# Run YOLO and list detected objects
def yolo_eval_and_list(yolov8_app, frame_pil, input_size):
    frame_resized, _, _ = pil_resize_pad(frame_pil, input_size)
    boxes, scores, class_idxs = yolov8_app.predict_boxes_from_image(frame_resized, raw_output=True)

    # Print object list with confidence
    print("\n🧾 Objects Detected:")
    for i in range(len(class_idxs[0])):
        cls_idx = class_idxs[0][i].item()
        score = round(scores[0][i].item(), 2)
        name = yolo_names.get(cls_idx, f"class_{cls_idx}")
        print(f" - {name} (Confidence: {score})")

# Main loop
def main_loop(yolov8_app, qhealth_config: QHealthConfig, is_debug=True):
    yolov8_h, yolov8_w = YoloV8Detector.get_input_spec()["image"][0][2:]
    frame_pil, _ = get_frame(qhealth_config, use_camera=True)
    if frame_pil is None:
        print("🚫 No camera frame captured.")
        return
    yolo_eval_and_list(yolov8_app, frame_pil, (yolov8_h, yolov8_w))

# Run program
if __name__ == "__main__":
    is_debug = True
    config_file = "./config.json"
    qhealth_config = QHealthConfig.from_config_file(config_file)

    if not qhealth_config.enable_degree0_cam:
        print("⚠️ No camera enabled. Exiting.")
    else:
        if is_debug:
            qhealth_config.setup_debug_figure()

        yolov8_app = load_yolov8_app(is_debug)
        app_scheduler = RepeatedTimer(
            qhealth_config.update_interval_s,
            main_loop, yolov8_app, qhealth_config, is_debug
        )

        print(f"🕒 Running for {qhealth_config.run_duration_s} seconds...")
        try:
            time.sleep(qhealth_config.run_duration_s)
        finally:
            app_scheduler.stop()