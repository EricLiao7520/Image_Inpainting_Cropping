
# -*- coding: utf-8 -*-
'''
Detector Only
'''
from .recognition import get_recognizer, get_text
from .utils import group_text_box, get_image_list, calculate_md5, get_paragraph,\
                   download_and_unzip, printProgressBar, diff, reformat_input,\
                   make_rotated_img_list, set_result_with_confidence,\
                   reformat_input_batched, merge_to_free
from .config import *
from bidi import get_display
import numpy as np
import cv2
import torch
import os
import sys
from PIL import Image
from logging import getLogger
import yaml
import json

LOGGER = getLogger(__name__)
if sys.version_info[0] == 2:
    from io import open
    from six.moves.urllib.request import urlretrieve
    from pathlib2 import Path
else:
    from urllib.request import urlretrieve
    from pathlib import Path

class Reader(object):
    """
    Minimal EasyOCR-like Reader that ONLY performs detection of text bounding boxes.
    Recognition functionality has been removed.
    """

    def __init__(self,
                 gpu=True,
                 model_storage_directory=None,
                 user_network_directory=None,
                 detect_network="craft",
                 download_enabled=True,
                 detector=True,
                 verbose=True,
                 quantize=True,
                 cudnn_benchmark=False):
        """
        Create a detection-only Reader.

        Parameters:
            gpu (bool): Enable GPU or not (default True).
            model_storage_directory (str): Custom path to store/load detection models.
            user_network_directory (str): Custom path for user-provided detection networks (rarely used).
            detect_network (str): Which detection network to use ("craft" or "dbnet18").
            download_enabled (bool): Whether to allow downloading missing models (default True).
            detector (bool): Whether to initialize the detector (default True).
            verbose (bool): Controls logging warnings/info.
            quantize (bool): Whether to use quantized models if supported.
            cudnn_benchmark (bool): Whether to enable cudnn benchmark for performance.
        """
        self.verbose = verbose
        self.download_enabled = download_enabled

        # --- Model storage directories ---
        self.model_storage_directory = MODULE_PATH + '/model'
        if model_storage_directory:
            self.model_storage_directory = model_storage_directory
        Path(self.model_storage_directory).mkdir(parents=True, exist_ok=True)

        self.user_network_directory = MODULE_PATH + '/user_network'
        if user_network_directory:
            self.user_network_directory = user_network_directory
        Path(self.user_network_directory).mkdir(parents=True, exist_ok=True)
        sys.path.append(self.user_network_directory)

        # --- Device selection ---
        if gpu is False:
            self.device = 'cpu'
            if verbose:
                LOGGER.warning('Using CPU. Note: This module is much faster with a GPU.')
        elif gpu is True:
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
                if verbose:
                    LOGGER.warning(
                        'Neither CUDA nor MPS are available - defaulting to CPU. '
                        'Note: This module is much faster with a GPU.'
                    )
        else:
            # Allow passing a custom device string
            self.device = gpu

        self.detection_models = detection_models
        self.quantize = quantize
        self.cudnn_benchmark = cudnn_benchmark
        self.support_detection_network = ['craft', 'dbnet18']

        if detector:
            detector_path = self.getDetectorPath(detect_network)
            self.detector = self.initDetector(detector_path) 

    def getDetectorPath(self, detect_network):
        if detect_network in self.support_detection_network:
            self.detect_network = detect_network
            if self.detect_network == 'craft':
                from .detection import get_detector, get_textbox
            elif self.detect_network in ['dbnet18']:
                from .detection_db import get_detector, get_textbox
            else:
                raise RuntimeError("Unsupport detector network. Support networks are craft and dbnet18.")
            self.get_textbox = get_textbox
            self.get_detector = get_detector
            corrupt_msg = 'MD5 hash mismatch, possible file corruption'
            detector_path = os.path.join(self.model_storage_directory, self.detection_models[self.detect_network]['filename'])
            if os.path.isfile(detector_path) == False:
                if not self.download_enabled:
                    raise FileNotFoundError("Missing %s and downloads disabled" % detector_path)
                LOGGER.warning('Downloading detection model, please wait. '
                               'This may take several minutes depending upon your network connection.')
                download_and_unzip(self.detection_models[self.detect_network]['url'], self.detection_models[self.detect_network]['filename'], self.model_storage_directory, self.verbose)
                assert calculate_md5(detector_path) == self.detection_models[self.detect_network]['md5sum'], corrupt_msg
                LOGGER.info('Download complete')
            elif calculate_md5(detector_path) != self.detection_models[self.detect_network]['md5sum']:
                if not self.download_enabled:
                    raise FileNotFoundError("MD5 mismatch for %s and downloads disabled" % detector_path)
                LOGGER.warning(corrupt_msg)
                os.remove(detector_path)
                LOGGER.warning('Re-downloading the detection model, please wait. '
                               'This may take several minutes depending upon your network connection.')
                download_and_unzip(self.detection_models[self.detect_network]['url'], self.detection_models[self.detect_network]['filename'], self.model_storage_directory, self.verbose)
                assert calculate_md5(detector_path) == self.detection_models[self.detect_network]['md5sum'], corrupt_msg
        else:
            raise RuntimeError("Unsupport detector network. Support networks are {}.".format(', '.join(self.support_detection_network)))
        
        return detector_path

    def initDetector(self, detector_path):
        return self.get_detector(detector_path, 
                                 device = self.device, 
                                 quantize = self.quantize, 
                                 cudnn_benchmark = self.cudnn_benchmark
                                 )
    

    def detect(self, img, min_size = 20, text_threshold = 0.7, low_text = 0.4,\
               link_threshold = 0.4,canvas_size = 2560, mag_ratio = 1.,\
               slope_ths = 0.1, ycenter_ths = 0.5, height_ths = 0.5,\
               width_ths = 0.5, add_margin = 0.1, reformat=True, optimal_num_chars=None,
               threshold = 0.2, bbox_min_score = 0.2, bbox_min_size = 3, max_candidates = 0,
               ):

        if reformat:
            img, img_cv_grey = reformat_input(img)

        text_box_list = self.get_textbox(self.detector, 
                                    img, 
                                    canvas_size = canvas_size, 
                                    mag_ratio = mag_ratio,
                                    text_threshold = text_threshold, 
                                    link_threshold = link_threshold, 
                                    low_text = low_text,
                                    poly = False, 
                                    device = self.device, 
                                    optimal_num_chars = optimal_num_chars,
                                    threshold = threshold, 
                                    bbox_min_score = bbox_min_score, 
                                    bbox_min_size = bbox_min_size, 
                                    max_candidates = max_candidates,
                                    )

        horizontal_list_agg, free_list_agg = [], []
        for text_box in text_box_list:
            horizontal_list, free_list = group_text_box(text_box, slope_ths,
                                                        ycenter_ths, height_ths,
                                                        width_ths, add_margin,
                                                        (optimal_num_chars is None))
            if min_size:
                horizontal_list = [i for i in horizontal_list if max(
                    i[1] - i[0], i[3] - i[2]) > min_size]
                free_list = [i for i in free_list if max(
                    diff([c[0] for c in i]), diff([c[1] for c in i])) > min_size]
            horizontal_list_agg.append(horizontal_list)
            free_list_agg.append(free_list)
        
        return horizontal_list_agg, free_list_agg

    def readtext(self,
                 image,
                 min_size=20,
                 text_threshold=0.7,
                 low_text=0.4,
                 link_threshold=0.4,
                 canvas_size=2560,
                 mag_ratio=1.,
                 slope_ths=0.1,
                 ycenter_ths=0.5,
                 height_ths=0.5,
                 width_ths=0.5,
                 add_margin=0.1,
                 threshold=0.2,
                 bbox_min_score=0.2,
                 bbox_min_size=3,
                 max_candidates=0):
        """
        Detect text bounding boxes (horizontal rectangles + free-form polygons).
        This method NO LONGER performs text recognition.

        Returns:
            (horizontal_list, free_list) for the first frame of detected boxes.
            Each is a list of bounding boxes:
              - horizontal boxes in [x1, x2, y1, y2] format
              - free boxes as polygon coordinates [[x1, y1], [x2, y2], ...]
        """
        # Perform detection
        horizontal_list_all, free_list_all = self.detect(
            image,
            min_size=min_size,
            text_threshold=text_threshold,
            low_text=low_text,
            link_threshold=link_threshold,
            canvas_size=canvas_size,
            mag_ratio=mag_ratio,
            slope_ths=slope_ths,
            ycenter_ths=ycenter_ths,
            height_ths=height_ths,
            width_ths=width_ths,
            add_margin=add_margin,
            threshold=threshold,
            bbox_min_score=bbox_min_score,
            bbox_min_size=bbox_min_size,
            max_candidates=max_candidates,
            reformat=True
        )
        result = []
        # The detect() method always returns lists with depth=1 or more.
        # Typically, we only need the first sub-list if you're processing one image.
        horizontal_list = horizontal_list_all[0] if horizontal_list_all else []
        free_list = free_list_all[0] if free_list_all else []
        return  horizontal_list, free_list