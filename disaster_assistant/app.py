import warnings
import time
import numpy as np

from qai_hub_models.models.yolov8_det.model import YoloV8Detector

from qai_hub_models.utils.image_processing import pil_resize_pad, pil_undo_resize_pad

from yolo import load_yolov8_app, yolo_eval_and_list
from utils import get_frame, QHealthConfig

import qai_hub as hub
target_device = hub.Device("Snapdragon X Elite CRD")
from enum import Enum

advice_eq = {
    "0" : "Stay calm and remember not to panic! Follow proper safety protocols and be aware of your surroundings :)",
    "26": "The handbag with a utility to carry important small objects (bandaids, medication, etc.) during crisis.",
    "33": "The kite will have a string / rope that can act as a knot to slow blood loss accompanied with clothing.",
    "39": "A bottle will be important to hydrate yourself in remote locations.",
    "41": "A cup can act as a small container to organize different materials.",
    "45": "A bowl can act as a small container to organize different materials.",
    "57": "Check if the couch has cushions, which can be an effective way to shelter yourself in times of earthquake or can provide fabric.",
    "60": "Please shelter under the dining table to protect your heads and shoulders from earthquake.",
    "67": "Finding a place with higher population might be better chance to connect cell phones to internet.",
    "72": "Avoid the refrigerator and other large objects.",
    "74": "A clock can keep track of time at times of disasters, especially if power is lost, or the batteries can be removed to power other more essential devices.",
    "76": "Use scissors to open packages of food and cut strips of fabric in the event of injury.",
    "43": "A knife can be used for self-defense, opening packages or cans, or cutting fabric to dress wounds.",
    "59": "A bed can be used to take shelter during an earthquake.",
    "65": "A remote may contain batteries that can be used to power more essential equipment.",
}

advice_flood = {
    "0" : "Stay calm and remember not to panic! Follow proper safety protocols and be aware of your surroundings :)",
    "26": "The handbag with a utility to carry important small objects (bandaids, medication, etc.) during crisis.",
    "33": "The kite will have a string / rope that can act as a knot to slow blood loss accompanied with clothing.",
    "39": "A bottle will be important to hydrate yourself. Keep water with you at all times.",
    "41": "A cup can act as a small container to store food or water.",
    "45": "A bowl can act as a small container to store food or water.",
    "57": "Check if the couch has cushions, which can provide fabric for first aid.",
    "67": "Finding a place with higher population might be better chance to connect cell phones to internet.",
    "74": "A clock can keep track of time at times of disasters, especially if power is lost, or the batteries can be removed to power other more essential devices.",
    "76": "Use scissors to open packages of food and cut strips of fabric in the event of injury.",
    "43": "A knife can be used for self-defense, opening packages or cans, or cutting fabric to dress wounds.",
    "59": "A bed can be used to protect yourself from an earthquake, block off windows and doors, or elevating yourself during flooding.",
    "65": "A remote may contain batteries that can be used to power more essential equipment.",
}


advice_fire = {
    "0" : "Stay calm and remember not to panic! Follow proper safety protocols and be aware of your surroundings :)",
    "26": "The handbag with a utility to carry important small objects (bandaids, medication, etc.) during crisis.",
    "33": "The kite will have a string / rope that can act as a knot to slow blood loss accompanied with clothing.",
    "39": "A bottle will be important to hydrate yourself in remote locations.",
    "41": "A cup can act as a small container to organize different materials.",
    "45": "A bowl can act as a small container to organize different materials.",
    "57": "If the couch is made of fabric, you can soak it in water and use it to protect yourself from smoke",
    "67": "Finding a place with higher population might be better chance to connect cell phones to internet.",
    "74": "A clock can keep track of time at times of disasters, especially if power is lost, or the batteries can be removed to power other more essential devices.",
    "76": "Use scissors to open packages of food and cut strips of fabric in the event of injury.",
    "43": "A knife can be used for self-defense, opening packages or cans, or cutting fabric to dress wounds.",
    "59": "A bed can be used to take shelter during an earthquake.",
    "65": "A remote may contain batteries that can be used to power more essential equipment.",
}


advice_hurricane = {
    "0" : "Stay calm and remember not to panic! Follow proper safety protocols and be aware of your surroundings :)",
    "26": "The handbag with a utility to carry important small objects (bandaids, medication, etc.) during crisis.",
    "33": "The kite will have a string / rope that can act as a knot to slow blood loss accompanied with clothing.",
    "39": "A bottle will be important to hydrate yourself in remote locations.",
    "41": "A cup can act as a small container to organize different materials.",
    "45": "A bowl can act as a small container to organize different materials.",
    "57": "Check if the couch has cushions, which can be an effective way to shelter yourself in times of earthquake or can provide fabric.",
    "60": "Use the dining table to block doors and windows.",
    "67": "Finding a place with higher population might be better chance to connect cell phones to internet.",
    "74": "A clock can keep track of time at times of disasters, especially if power is lost, or the batteries can be removed to power other more essential devices.",
    "76": "Use scissors to open packages of food and cut strips of fabric in the event of injury.",
    "43": "A knife can be used for self-defense, opening packages or cans, or cutting fabric to dress wounds.",
    "59": "A bed can be used to take shelter during an earthquake.",
    "65": "A remote may contain batteries that can be used to power more essential equipment.",
    "72": "Use the refrigerator to block windows or doors."
}

advice = [advice_eq, advice_flood, advice_fire, advice_hurricane]

# Example usage
def photo(disaster):
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

    objects = yolo_eval_and_list(yolov8_app, frame_pil, (yolov8_h, yolov8_w))
    already_listed = set()
    people = 0
    str_out = []
    for i in objects:
        if i == 0:
            people+=1
        if str(i) in advice[disaster] and i not in already_listed:
            str_out.append(advice[disaster][str(i)])
        already_listed.add(i)
    if people > 1:
        str_out.append("Multiple people detected. Stay calm, stay together, and support one another — safety is stronger in numbers.")
    return str_out