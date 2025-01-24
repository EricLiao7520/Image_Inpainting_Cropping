import os
import sys
import glob
import cv2
import numpy as np
from functools import partial
from util import find_bounding_boxes_from_mask, merge_overlapping_boxes
from concurrent.futures import ProcessPoolExecutor, as_completed

def crop_frame(
    frame_path:str,
    cropped_boundaries:list,
    cropped_folder:str
    ):
    '''
    Crop a frame based on specified boundaries and save it to the cropped folder.

    params:
        frame_path (str): Path of the frame folder
        cropped_boundaries (list): List of the boundaries[x, y, w, h] for cropping
        cropped_folder: Folder to saving the cropped frames
    '''
    frame_file = os.path.basename(frame_path)
    image = cv2.imread(frame_path)
    cropped_image = image[cropped_boundaries[0]:cropped_boundaries[1], 0:cropped_boundaries[2]]
    cropped_path = os.path.join(cropped_folder, frame_file)
    cv2.imwrite(cropped_path, cropped_image)

def get_boundaries(
    cropped_boundaries:dict,
    frame_width: int,
    frame_height: int):
    '''
    Calculate cropping boundaries based on the provided top and bottom thresholds.

    Params:
        cropped_boundaries: Dictionary with 'top' and 'bottom' cropping thresholds.
        frame_width: Width of the video frames
        frame_height: Height of the video frames
    Returns:
        List of the cropping boundaries, with the format [height, width]
    '''
    y_start = 0
    y_end = frame_height
    if 'top' in cropped_boundaries:
        y_max = int(cropped_boundaries['top'])
        if y_max < y_end:
            y_start = y_max
    if 'bottom' in cropped_boundaries:
        y_min = int(cropped_boundaries['bottom'])
        if y_min > y_start:
            y_end = y_min
    return [y_start, y_end, frame_width]

def crop_frames(frames_folder, mask_folder, cropped_folder):
    """
    Process video frames and masks to identify and merge bounding boxes for cropping.

    Parameters:
    - frames_folder: Path to the folder which containing video frames
    - mask_folder: Path to the folder which containing mask images
    - cropped_folder: Path to save cropped images

    Returns:
    - list: List of cropped images or bounding boxes as needed.
    """
    print("Starting video cropping process.")

    #initialization
    if not os.path.exists(cropped_folder):
         os.makedirs(cropped_folder, exist_ok=True)

    #Retrieve and sort mask file paths
    mask_paths = sorted(glob.glob(os.path.join(mask_folder, "*_mask.png")))

    if not mask_paths:
        print("No mask files found.")
        return []
    
    all_bounding_boxes = []

    # Process each mask to extract bounding boxes
    for idx, mask_path in enumerate(mask_paths):
        frame_file =os.path.basename(mask_path)
        if not os.path.exists(mask_path):
            print(f"Frame {frame_file} does not exist. Skipping.")
            continue
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        boxes = find_bounding_boxes_from_mask(mask, crop=True) #(x, y, w, h)
        all_bounding_boxes.extend(boxes)

    #Getting frame dimensions
    frame_sample_path = mask_paths[0]
    sample_image = cv2.imread(frame_sample_path)
    if sample_image is None:
        print(f"Failed to read sample frame {frame_sample_path}. Cannot determine frame dimensions.")
        return
    frame_height, frame_width = sample_image.shape[:2]
    categorized_boxes = {'top': [], 'bottom': []}
    thr_top = 0.3 * frame_height
    thr_bottom = (1 - 0.3) * frame_height

    # Decide subtitle position based on median y position
    for box in all_bounding_boxes:
        #print(box)
        center_y = box[1] + box[3] / 2
        if center_y < thr_top:
            categorized_boxes['top'].append(box)
        elif center_y  > thr_bottom:
            categorized_boxes['bottom'].append(box)
            
    #Define the cropping boundary based on the subtitle position
    cropped_boundaries = {}
    if categorized_boxes['top']:
        # For top subtitles, crop up to the maximum y_max of top boxes
        y_max_top = max([y + h for (x, y, w, h) in categorized_boxes['top']])
        cropped_boundaries['top'] = y_max_top
        print(f"Global top cropping boundary set at y={y_max_top}")

    if categorized_boxes['bottom']:
        # For bottom subtitles, crop from the minimum y_min of bottom boxes
        y_min_bottom = min([y for (x, y, w, h) in categorized_boxes['bottom']])
        cropped_boundaries['bottom'] = y_min_bottom
        print(f"Global bottom cropping boundary set at y={y_min_bottom}")
    frame_pattern = os.path.join(frames_folder, "frame_????.png")  # Matches exactly four characters
    frame_paths = sorted(glob.glob(frame_pattern))
    #get boundaries
    boundaries = get_boundaries(cropped_boundaries, frame_width, frame_height)
    #Crop subtitle from every frames
    crop_partial = partial(
        crop_frame,  
        cropped_boundaries= boundaries,
        cropped_folder=cropped_folder
    )
    # Process frames in parallel to speed up by using ProcessPoolExecutor
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(crop_partial, image_path): image_path for image_path in frame_paths}
        for future in as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"Error processing frame {futures[future]}: {e}")
    print("Done Video cropping")