import sys
sys.path.append('../camera-control')

import os

from vggt_detect.Module.detector import Detector


def demo():
    home = os.environ['HOME']
    model_file_path = home + '/chLi/Model/VGGT/VGGT-1B/model.pt'
    device = 'cuda:0'
    image_folder_path = home + '/chLi/Dataset/GS/haizei_1/input/images/'
    mode = 'pad'
    robust_mode = True
    cos_thresh = 0.95

    detector = Detector(model_file_path, device)

    predictions = detector.detectImageFolder(
        image_folder_path,
        mode,
        robust_mode,
        cos_thresh,
    )
    assert predictions is not None

    print(predictions.keys())
    return True
