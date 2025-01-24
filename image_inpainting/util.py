import numpy as np
import cv2
import os
import sys
import subprocess
import easyocr
import shutil
from collections import defaultdict

class UnionFind:
    '''Disjoint Set Union data structure to handle merging of bounding boxes'''
    def __init__(self, size):
        self.parent = list(range(size))
    def find(self, i):
        if self.parent[i] == i:
            return i
        return self.find(self.parent[i])

    def union(self, i, j):
        pi, pj = self.find(i), self.find(j)
        if(pi != pj):
            self.parent[pi] = pj

def calculate_iou(boxes1, boxes2):
    '''
    Calculate Intersection over union(IoU) between two sets of bounding boxes.

    Params:
        boxes1: First set of bounding boxes
        boxes2: Second set of bounding boxes
    Return:
        iou: Array of IoU values between each pair of boxes
    '''
    boxes1 = np.array(boxes1, dtype=float).reshape(-1, 4)  # Shape: (N, 4)
    boxes2 = np.array(boxes2, dtype=float).reshape(-1, 4)  # Shape: (M, 4)

    #Expand dimensions for broadcasting
    boxes1 = np.expand_dims(boxes1, 1)  # Shape: (N, 1, 4)
    boxes2 = np.expand_dims(boxes2, 0)  # Shape: (1, M, 4)


    inter_min = np.maximum(boxes1[..., :2],  boxes2[..., :2])  # (N, M, 2)
    xi2 = np.minimum(boxes1[..., 0] + boxes1[..., 2], boxes2[..., 0] + boxes2[..., 2])
    yi2 = np.minimum(boxes1[..., 1] + boxes1[..., 3], boxes2[..., 1] + boxes2[..., 3])
    inter_max = np.stack([xi2, yi2], axis=-1)  # (N, M, 2)

    inter = np.clip(inter_max - inter_min, a_min=0, a_max=None)
    inter_area = inter[..., 0] * inter[..., 1]  # (N, M)

    area1 = boxes1[..., 2] * boxes1[..., 3]  # (N, 1)
    area2 = boxes2[..., 2] * boxes2[..., 3]  # (1, M)
    union_area = area1 + area2 - inter_area

    return np.divide(inter_area, np.maximum(union_area, 1e-10))

def merge_overlapping_boxes(boxes, overlap_threshold=0.05):
    """
    Merge overlapping bounding boxes based on IoU threshold.

    Parameters:
        boxes:                List of bounding boxes [x, y, width, height].
        overlap_threshold:    IoU threshold to consider boxes as overlapping.

    Returns:
        list:                 Merged bounding boxes (x, y, width, height).
    """
    if not boxes:
        return []
    
    boxes = np.array(boxes, dtype=float)
    n = len(boxes)
    uf = UnionFind(n)

    for i in range(n):
        for j in range(i + 1, n):
            iou = calculate_iou(boxes[i], boxes[j])
            if iou > overlap_threshold:
                uf.union(i, j)
    
    cluster = defaultdict(list)
    for i in range(n):
        parent = uf.find(i)
        cluster[parent].append(i)

    merged = []

    #merge boxes in each clusters
    for indices in cluster.values():
        selected_box = boxes[indices]
        x_min = selected_box[:, 0].min()
        y_min = selected_box[:, 1].min()
        x_max = (selected_box[:, 0] + selected_box[:, 2]).max()
        y_max = (selected_box[:, 1] + selected_box[:, 3]).max()
        merged.append((x_min, y_min, x_max - x_min, y_max - y_min))

    return merged

def pad_to_proper_size(image, expected_H = 896, expected_W = 640):
    '''
    Reize image to enhance the OCR detection.

    Params:
        image:      The input image to be resized
    
    Return: 
        resized:    the resized image
        scale:      The scaling factor
        expected_H: Expected height after scaling 
        expected_W: Expected width after scaling 
    '''

    h, w = image.shape[:2]
    proper_limit = h + w
    if proper_limit > expected_H + expected_W:
        scale_long = expected_H / max(h, w)
        scale_short = expected_W / min(h, w)
        scale = min(scale_long, scale_short)

        if scale < 1.0:
            #need to downsize the image
            new_w = int(round(w * scale))
            new_h = int(round(h * scale))
            resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            return resized, scale

    return image, 1.0

def pad_to_64(image: np.ndarray):
    '''
    Adding zero padding to ensure that image dimensions(neareast multiple of 64) 
    are compatible with the model input.

    Params:
        image       : The input image.

    Returns: 
        padded_image: The padded Image
        padded_dims : The new dimensions of padded image(width, height)
        offset      : The padding offset(y, x)

    '''
    original_height, original_width = image.shape[:2]
    padded_height = ((original_height + 63) // 64) * 64
    padded_width = ((original_width + 63) // 64) * 64
    offset_y = (padded_height - original_height) // 2
    offset_x = (padded_width - original_width) // 2

    if len(image.shape) == 3:
    # Color image
        padded_image = np.zeros((padded_height, padded_width, image.shape[2]), 
                            dtype=image.dtype)
    else:
    # Grayscale image
        padded_image = np.zeros((padded_height, padded_width), dtype=image.dtype)
        
    padded_image[offset_y:offset_y+original_height, 
                offset_x:offset_x+original_width] = image

    return padded_image, (padded_width, padded_height), (offset_y, offset_x)

def align_bbox_to_64(box, W, H, pad_factor = 8, expected_resolution = 720, aspect_ratio = 1.3):
    """
    Aligns the bounding box so that its width and height are multiples of 64.

    Params: 
        box :                   The bounding box (x1, y1, x2, y2)
        W :                     The width of the image
        H :                     The height of the image
        pad_factor :            The padding factor
        expected_resolution:    use to adjust the padding_factor for high resolution to get more information
        aspect_ratio:           Use to adjust the ratio of height and width of bounding boxes
    """
    x1, y1, x2, y2  = box

    #current image size
    width = (x2 - x1 + 1)
    height = (y2 - y1 + 1)

    stride_h = 64 * (pad_factor + round(max(H , W) / expected_resolution))
    stride_w = stride_h * aspect_ratio

    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    new_h = ((height + stride_h - 1) // stride_h) * stride_h
    new_w = ((width + stride_w - 1) // stride_w) * stride_w

    new_x1 = cx - (new_w) // 2
    new_y1 = cy - (new_h) // 2
    new_x2 = new_x1 + new_w - 1
    new_y2 = new_y1 + new_h - 1
     # Adjust new_x1 and new_x2 to ensure the box fits within [0, W-1]

    if new_x1 < 0:
        new_x2 = new_w - 1
    if new_x2 >= W:
        new_x2 = W - 1
        new_x1 = W - new_w

    new_x1 = max(0, new_x1)
    new_y1 = max(0, new_y1)
    new_x2 = min(W - 1, new_x2)
    new_y2 = min(H - 1, new_y2)
    return new_x1, new_y1, new_x2, new_y2

def unpad_to_original(
    padded_image, 
    original_size, 
    offset = None,
    scale = 1.0,
):
    """
    Crop out the original (H, W) region from a padded image based on offset.
    
    Parameters:
        padded_image :      The inpainted padded image, shape [Hp, Wp] or [Hp, Wp, C].
        original_size :     (original_height, original_width).
        offset :            (offset_y, offset_x) from pad_to_64.
        scale :             Scaling factor applied during padding
    
    Returns:
        unpadded_image:    Cropped image of original size
    """
    (original_height, original_width) = original_size
    #apply offset_based cropping if offset is provided
    if offset:
        (offset_y, offset_x) = offset
        # Slice out the original region
        unpadded_image = crop_with_offset(padded_image, offset_y, offset_x, original_height, original_width)

    #resize if scale is not 1.0    
    if scale != 1.0:

        unpadded_image = cv2.resize(
            padded_image,
            (original_height, original_width),
             interpolation = cv2.INTER_LINEAR
        )

    return unpadded_image

def crop_with_offset(image, offset_y, offset_x, original_height, original_width):
    '''
    Helper function to crop the image based on the provided offset.
    '''
    if len(image.shape) == 3:
        return image[offset_y:offset_y + original_height, offset_x:offset_x + original_width, :]
    return image[offset_y:offset_y + original_height, offset_x:offset_x + original_width]  # Grayscale

def run_command(command, description):
    '''
    Excutes a system command

    Params:
        command: List of command arguments
        description: Description of the command for logging
    '''

    print(f"Starting: {description}")
    try:
        subprocess.run(command, check=True)
        print(f"Completed: {description}\n")
    except subprocess.CalledProcessError as e:
        print(f"Error during: {description}")
        sys.exit(1)

def inpaint_frames(frames_folder, inpainted_dir, steps):
    '''
    Call model for frames inpainting.

    Params:
        frames_folder: Path to the frames folder
        inpainted_dir: The Directory where the inpainted result will be saved
        steps: The steps of DDIM samplier

    '''
    inpaint_command = [
        "python3", "../inpaint/scripts/roi_inpaint.py",
        "--indir", frames_folder,
        "--outdir", inpainted_dir,
        "--steps", str(steps)
    ]

    run_command(inpaint_command, "Inpainting Frames")
    
def extract_audio(video_path, audio_path):
    '''
    Extracts audio from the given video and saves it as an AAC file

    Params: 
        video_path:     Path to the input video file
        audio_path:     The directory where the audio file will be saved
    '''
    if not os.path.exists(audio_path):
        os.makedirs(audio_path)
    video_filename = os.path.basename(video_path)       # e.g., "my_video.mp4"
    video_root = os.path.splitext(video_filename)[0]    # e.g., ("my_video")
    audio_filename = video_root + ".aac"                # e.g., "my_video.aac"
    audio_path = os.path.join(audio_path, audio_filename)

    command = [
        "ffmpeg", "-y",   
        "-i", video_path,
        "-vn",
        "-c:a", "copy",
        audio_path
    ]

    run_command(command, "Extracting Audio from Source Video")

def combine_frames_and_audio(frame_path, audio_path, output_video, frame_rate):
    '''
    Combine video frames and audio to generate a video

    Params:
        frame_path: Path to the frame images.
        audio_path: Path to the audio file
        output_video: Path to save the combined video
        frame_rate: The fps of the original video
    '''
    command = [
        "ffmpeg", "-y",                      
        "-framerate", str(frame_rate),       
        "-i", frame_path,              
        "-i", audio_path,                   
        "-c:v", "copy", # No compression              
        "-pix_fmt", "yuv420p",               
        "-c:a", "copy",                   
        output_video
    ]

    run_command(command, "Combining Processed Frames and Audio")

def run_ocr_on_frame(frame, pad, reader, H, W, scale=1.0, thr_top_ratio = 0.2):
    '''
    Run OCR on a frame, scale the bounding boxes, and apply padding.

    Params:
        frame:  The input frame
        pad: Padding to apply on bounding box
        reader: OCR reader
        H: Height of the original image
        W: Width of the original image
        scale:  Scaling factor

    Returns:
        ocr_bboxes: List of bounding boxes for detected texts.
    '''

    horizontal_results,_ = reader.readtext(frame)
    ocr_bboxes = []
    
    thr_top = thr_top_ratio *  H
    thr_bottom = (1 - thr_top_ratio) * H
    for bbox in horizontal_results:
        x1, x2, y1, y2 = bbox
        if scale != 0:
            x1, y1, x2, y2 = apply_scale(x1, y1, x2, y2, scale)

        center_y = (y1 + y2) / 2.0
        if thr_top < center_y < thr_bottom:
            continue
        
        padded_x1, padded_y1, padded_x2, padded_y2 = apply_padding(x1, y1, x2, y2, pad, W, H)
        ocr_bboxes.append([padded_x1, padded_y1, padded_x2,padded_y2])
    return ocr_bboxes

def apply_scale(x1, y1, x2, y2, scale):
    '''
    Helper method for scaling bouding box for the original size of image.
    '''

    x1 = int(np.floor(x1 / scale))
    y1 = int(np.floor(y1 / scale))
    x2 = int(np.ceil(x2 / scale))
    y2 = int(np.ceil(y2 / scale))
    return x1, y1, x2, y2

def apply_padding(x1, y1, x2, y2, pad, W, H):
    '''
    Helper method for applying padding to bounding box and ensures it's within image bounds.
    '''

    padded_x1 = max(int(x1 - pad), 0)
    padded_y1 = max(int(y1 - pad), 0)
    padded_x2 = min(int(x2 + pad), W - 1)
    padded_y2 = min(int(y2 + pad), H - 1)
    return padded_x1, padded_y1, padded_x2, padded_y2

def ocr_padded(frames_folder, processed_dir, pad = 10,  crop=False, sample_ratio = 0.5, min_appear_percentage = 0.2):
    '''
    Do OCR on padded frames and increase OCR accuracy by clustering bounding boxes.

    Parameters:
        frames_folder:          Directory of Original frames.
        processed_dir:          Directory to save processed frames
        pad:                    Padding pixels to apply on bounding boxes 
        crop :                  Whether to apply padding to frames before OCR.
        sample_ratio:           Ratio of frames to sample for OCR
        min_appear_percentage:  Minimum percentage of appearness for bouding box filtering.
    Returns:
        frame_list:             List of frame data
        frame_file_list         List of frames file name
        ocr_list:               Bounding box for each frames.[x1, x2, y1, y2]
    '''
    reader = easyocr.Reader()
    frame_files = get_sorted_frame_files(frames_folder)
    num_samples = max(1, int(len(frame_files)  * sample_ratio))  # Ensure at least one frame is selected
    sampled_indices = np.random.choice(len(frame_files), size=num_samples, replace=False)
    sample_list = [frame_files[i] for i in sampled_indices]
    remain_list = [frame_files[i] for i in range(len(frame_files)) if i not in sampled_indices]

    first_pass_list, second_pass_list = determine_frame_passes(sample_list, remain_list, crop)
    ocr_dict = {}
    frame_data_list = {}

    ocr_dict, frame_data_list = process_frames(frames_folder, processed_dir, first_pass_list, ocr_dict, frame_data_list, reader, pad)
    ocr_dict = cluster_and_filter(ocr_dict, 
                min_appear = determine_min_appearance(len(frame_data_list), min_appear_percentage))
    if not crop and second_pass_list:
        ocr_dict, frame_data_list = process_frames(frames_folder, processed_dir, second_pass_list, ocr_dict, frame_data_list, reader, pad)
        ocr_dict = cluster_and_filter(ocr_dict, 
        min_appear = determine_min_appearance(len(frame_data_list), min_appear_percentage))  
    frame_list, frame_file_list, ocr_list = compile_final_ocr_results(frames_folder, processed_dir, frame_files, ocr_dict, frame_data_list)
    return frame_list, frame_file_list, ocr_list

def get_sorted_frame_files(frames_folder):
    '''
    Helper method for sorting image files in the frames folder
    '''
     
    return sorted(f for f in os.listdir(frames_folder) if f.lower().endswith('.png'))

def determine_frame_passes(sampled_frames, remained_Frames, crop):
    '''
    Helper method for determining first and second pass frames lists based on cropping.
    '''

    if crop:
        return sampled_frames, []
    return sampled_frames, remained_Frames

def process_frames(frames_folder, processed_dir, frame_list, ocr_dict, frame_data_list, reader, pad):
    '''
    Helper method for Processing OCR on each frames in the provided list
    '''
    frame_path = os.path.join(frames_folder, frame_list[0])
    frame = cv2.imread(frame_path)
    H, W = frame.shape[:2]

    for frame_file in frame_list:
        frame_path = os.path.join(frames_folder, frame_file)
        frame = cv2.imread(frame_path)
        resized_frame, scale = pad_to_proper_size(frame)
        bboxes = run_ocr_on_frame(resized_frame, pad, reader, H, W, scale)
        if not bboxes:
            handle_no_text_in_frame(frame_file, frames_folder, processed_dir)
        ocr_dict[frame_file] = bboxes
        frame_data_list[frame_file] = frame
    return ocr_dict, frame_data_list

def handle_no_text_in_frame(frame_file, frames_folder, processed_dir):
    """Handles the case where no text is detected in the frame."""
    print(f"Skipping {frame_file}, No text in this frame")
    original_frame_path = os.path.join(frames_folder, frame_file)
    inpainted_path = os.path.join(processed_dir, frame_file)
    try:
        shutil.copyfile(original_frame_path, inpainted_path)
    except FileNotFoundError:
        print("The source file does not exist.")

def determine_min_appearance(frame_data_len, min_appear_percentage, expected_len = 10):
    '''
    Helper method for Calculating the minimum appearance thresthod
    Params:
        frame_data_len: Total length of frame that contains bounding box
        expected_len: Expected Total length of frame that can be considered to ignore minor bounding boxes
    '''
    return 0 if frame_data_len < expected_len else int(frame_data_len * min_appear_percentage)

def compile_final_ocr_results(frames_folder, processed_dir, frame_files, ocr_dict, frame_data_list):
    '''
    Helper method for compiling the final OCR results after processing frames
    '''
    frame_list, frame_file_list, ocr_list = [], [], []
    for frame_file in frame_files:
        if frame_file not in ocr_dict or not ocr_dict[frame_file]:
            handle_no_text_in_frame(frame_file, frames_folder, processed_dir)
        frame_list.append(frame_data_list[frame_file])
        frame_file_list.append(frame_file)
        ocr_list.append(ocr_dict[frame_file])
    return frame_list,frame_file_list, ocr_list

def find_bounding_boxes_from_mask(mask: np.ndarray, min_area=1000, pad = 30, crop=False):
    '''
    Find bounding boxes around all white regions in the mask
    Params:
        mask:       An array for a mask
        min_area:   The minimum area that counts as a valid Contour
        pad:        The number of padding
    Return:
       boxes: a list of bounding boxes in the format (x, y, w, h)
    '''
    mask_bin = mask.astype(np.uint8)
    contours, _ = cv2.findContours(mask_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    H, W = mask.shape[:2]
    if crop:
        pad = 0
    for contour in contours:
        if cv2.contourArea(contour) < min_area:  # Filter out small regions
            continue
        x, y, w, h = cv2.boundingRect(contour)
        x1 = max(x - pad, 0)
        y1 = max(y - pad, 0)
        x2 = min(x + w + pad, W - 1)
        y2 = min(y + h + pad, H - 1)
        w1 = x2 - x1
        h1 = y2 - y1
        boxes.append([x1 ,y1, w1, h1])
    return boxes

def crop_with_box(image_np, mask_np, box):
    '''
    Crop both image and mask using the providing bounding box
    '''
    x1, y1, x2, y2 = map(int, box)
    # image_np is [H, W, 3], mask_np is [H, W]
    cropped_image = image_np[y1:y2+1, x1:x2+1, :]  
    cropped_mask  = mask_np[y1:y2+1, x1:x2+1]
    return cropped_image, cropped_mask

def paste_inpainted_batch(orig_image_np, batch_np, box):
    """
    Paste the inpainted patch_np back into orig_image_np at the specified region.
    The region is defined by the bounding box = (x1, y1, x2, y2).
    """
    x1, y1, x2, y2 = map(int, box)
    out = orig_image_np.copy()  # [H, W, 3]
    out[y1:y2+1, x1:x2+1, :] = batch_np
    return out


def cluster_and_filter(ocr_dict, iou_threshold = 0.05, min_appear=0, frame_window = 2):
    '''
    Clustering bounding boxes across multiple frames and filters based on appearance frequency.

    Params:
        ocr_dict: A dictionary with frame names as keys and list of bounding boxes as values.
        iou_threshold: Threshold for Intersection over Union(IoU)to consider boxes for overlapping.
        min_appear: Minimum number of frames a cluster must appear in to be considered valid.
        frame_window: Number of frames to look at on either side of the current frame for clustering.
    Returns:
         filter_dict: A dictionary with frame names as keys and list of bounding boxes as values.
    '''
    frames = sorted(ocr_dict.keys()) #Format: "0001.png"
    frame_to_idx = {f: i for i, f in enumerate(frames)}

    #Clustering all the boxes in frames.
    all_bboxes = []
    for f in frames:
        for box in ocr_dict[f]:
            all_bboxes.append({
                'frame_idx': frame_to_idx[f],
                'box' : box
            })

    #initialize UnonFind for clustering
    uf = UnionFind(len(all_bboxes))
    bboxes_by_frame = defaultdict(list)

    #Group bounding box indices by frame
    for i, info in enumerate(all_bboxes):
        frame_idx = info['frame_idx']
        bboxes_by_frame[frame_idx].append(i) #all bounding box for current frame(Ex: 1 - 5)

    boxes = np.array([bbox['box'] for bbox in all_bboxes])

    #CLustering bounding boxes based on IoU across nearby frames
    for f in bboxes_by_frame:
        indices_in_f = bboxes_by_frame[f]
        for i_idx in indices_in_f:# For bounding box in that frame
            box_i = boxes[i_idx]
            for offset in range(-frame_window, frame_window+ + 1): #Nearby frames
                nearby_frame = f + offset
                if nearby_frame not in bboxes_by_frame: #Skip out-of-bound frames
                    continue
                indices_nearby = bboxes_by_frame[nearby_frame]
                if len(indices_nearby) == 0:
                    continue

                boxes_j = boxes[indices_nearby]
                ious = calculate_iou(np.array([box_i]), boxes_j).flatten()
                valid = np.where(ious > iou_threshold)[0]
                
                for idx in valid:
                    j_idx = indices_nearby[idx]
                    if j_idx == i_idx:
                         continue #skip if comparing the box to it self
                    if offset == 0 and j_idx < i_idx:
                         continue #skip for double comparisons
                    uf.union(i_idx, j_idx)

    #Organize bounding boxes into clusters based on the UnionFind results
    cluster_dict = defaultdict(list)
    for i in range(len(all_bboxes)):
        parent = uf.find(i)
        cluster_dict[parent].append(i)

    #Filter clusters by the number of distinct frames they appear in
    final_clusters = defaultdict(list)
    for cid, indices in cluster_dict.items():
        distinct_frames = set(all_bboxes[idx]['frame_idx'] for idx in indices)

        if len(distinct_frames) < min_appear:
            continue
        
        #Add valid clusters to the final clusters dictionary
        for idx in indices:
            fidx = all_bboxes[idx]['frame_idx']
            box  = all_bboxes[idx]['box']
            final_clusters[fidx].append(box)
       
    filter_dict = {}
    for f in frames:
        fidx = frame_to_idx[f]
        filter_dict[f] = final_clusters.get(fidx, [])

    return filter_dict

def detectHorizontalBlackEdges(frame_path, threshold = 0.3):
    '''
    Detects the horizontal black edges in a given frame. It looks for region with black(0) 
    to define top and bottom boundaries for cropping

    Param:
        frame_path: Path to the frame
        threshold: area brightness threshold

    Return:
        Top and bottom row indices that define the horizontal cropping region
    '''
    image = cv2.imread(frame_path)
    h, w = image.shape[:2]
    first_col = image[:, 0]
    row_indices = np.nonzero(first_col)[0]

    if len(row_indices) == 0:
        # Entire first column is black
        return (0, h - 1)

    #Convert to grayscope image
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    total_brightness = np.sum(gray_image)
    max_brightness = h * w * 255 #Maximum brightness for a white image

    #If total brightness is less than 30%. assume the image is not good for edge detection
    if total_brightness < max_brightness * threshold:
        return (0, h - 1)

    # The top is the first non-black row; bottom is the last non-black row
    top = row_indices[0]
    bottom = row_indices[-1]
    return (top, bottom)

def sampleHorizontalEdges(frames_folder: str, 
    sample_ratio: float = 0.2
    ):
    '''
    Sampling a subset of frames to detect horizontal balck edges for cropping

    Params:
        frames_folder: The directory of frame images.
        sample_ratio: Ratio of frames to sample
    Return:
        Top and bottom row indices that define the global horizontal cropping region.
    '''
    frame_files = get_sorted_frame_files(frames_folder)
    first_path = os.path.join(frames_folder, frame_files[0])
    tmp_img = cv2.imread(first_path)
    h, w = tmp_img.shape[:2]
    global_top    = 0
    global_bottom = h - 1
    num_samples = max(1, int(len(frame_files)  * sample_ratio))  # Ensure at least one frame is selected
    sampled_indices = np.random.choice(len(frame_files), size=num_samples, replace=False)
    sample_list = [frame_files[i] for i in sampled_indices]

    for sample_frame in sample_list:
        fpath = os.path.join(frames_folder, sample_frame)
        tb = detectHorizontalBlackEdges(fpath)
        top_i, bottom_i = tb
        if top_i > global_top:
            global_top = top_i
        if bottom_i < global_bottom:
            global_bottom = bottom_i
    return (global_top, global_bottom)

def applyGlobalHorizontalCrop(frames_folder, top, bottom):
    '''
    Appling the global horizontal cropping to all frames in the specified folder

    Params:
        frames_folder: The directory of frame images.
        top: The top row index for cropping
        bottom: The bottom row index for cropping
    '''

    out_folder =  frames_folder
    frame_files = get_sorted_frame_files(frames_folder)
    for f in frame_files:
        inpath = os.path.join(frames_folder, f)
        outpath = os.path.join(out_folder, f)
        image = cv2.imread(inpath)
        h, w = image.shape[:2]
        cropped = image[top:bottom, 0:w]
        cv2.imwrite(outpath, cropped)

def letterBox_detection(frames_folder: str, 
    sample_interval: float = 0.2
    ):
    '''
    Detecting letterbox black edges in a set of frames and appling cropping based on sampled edges.
    
    Params:
        frames_folder: The directory of frame images.
        sample_interval: Fraction of frames to sample for detecting horizontal black edges.
    '''
    (top, bottom) = sampleHorizontalEdges(frames_folder, sample_interval)
    applyGlobalHorizontalCrop(frames_folder, top, bottom)