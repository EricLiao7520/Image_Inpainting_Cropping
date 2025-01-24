import cv2
import torch
import sys, os
import numpy as np
from PIL import Image
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

#Turn on TF32 support for newer CUDA devices
if torch.cuda.get_device_properties(0).major >= 8:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
CHECKPOINT = "../sam2/checkpoints/sam2.1_hiera_large.pt"
CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
BATCH_SIZE = 12

def generate_masks(mask_folder, frame_list, frame_file_list, ocr_list, crop=False):
    '''
    Generates masks for subtitles using SAM2 and bounding boxes.

    Params:
        mask_folder:        Directory to save the generated masks.
        frame_list:         List of image data for each frame
        frame_file_list:    List of file names for each frame
        ocr_list:           List of OCR bounding boxes for each frame
        crop:               Whether to apply subtitle cropping
    '''

    #Initialize SAM2 for inference
    predictor = SAM2ImagePredictor(build_sam2(CONFIG, CHECKPOINT, apply_postprocessing=False))

     # Batch the frames for processing
    batches =   [(frame_list[i : i + BATCH_SIZE], frame_file_list[i : i + BATCH_SIZE], ocr_list[i : i + BATCH_SIZE])
                for i in range(0, len(frame_list), BATCH_SIZE)]

    # process each batch
    for frame_batch, frame_file_batch, ocr_batch in batches:
        process_batch(predictor, frame_batch, frame_file_batch, ocr_batch, mask_folder, crop)
    
    #clean up
    del predictor
    torch.cuda.empty_cache()

def process_batch(predictor, image_batch, frame_file_batch, box_batch, mask_folder, crop):
    '''
    Process a batch of frames, generate masks, and save them to a folder.

    Params:
        predictor:          The SAM2 model used for inference
        image_batch:        A batch of frames
        frame_file_batch:   file names for each frame
        box_batch:          OCR bounding boxes for each frame
        mask_folder:        Directory to save the masks.
        crop:               Whether to apply subtitle cropping
    '''

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        #Set the image batch for prediction
        predictor.set_image_batch(image_batch)

        #Batch prediction
        masks_batch, _, _ = predictor.predict_batch(
            None,
            None,
            box_batch=box_batch,
            multimask_output=False
        )
    #Process and save the masks for each frame    
    for masks, frame_file in zip(masks_batch, frame_file_batch):
        if masks.shape[0] != 1:
            masks = masks.squeeze(1)
        
        #Combine the masks into a single mask
        combined_mask = np.any(masks, axis=0)
        # Convert to 0-255
        combined_mask = (combined_mask * 255).astype(np.uint8)
        
        #Apply morphological operations to refind the mask
        if not crop:
           combined_mask = refine_mask_edges(combined_mask)

        # Save the mask
        save_mask(mask_folder, frame_file, combined_mask)

def refine_mask_edges(mask):
    '''
    Apply morphological operations to refine the mask edges.

    Parameters:
        mask: The mask to refine.

    Returns:
        refined_mask: The refined mask.
    '''

    kernel_size = 5
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernel_size,kernel_size))
    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(kernel_size,kernel_size))
    # Erode and dilate to smooth edges and expand the mask area
    refined_mask = cv2.erode(mask, kernel, iterations=1) 
    refined_mask = cv2.dilate(refined_mask, kernel1, iterations=2)
    return refined_mask

def save_mask(mask_folder, frame_file, mask):
    '''
    Save the mask to the specified folder

    Params:
        mask_folder:        Directory to save the masks.
        frame_file:         file name for a frame
        mask:               The mask to save
    '''

    mask_fileName = frame_file.replace('.png', '_mask.png')
    mask_pil = Image.fromarray(mask).convert("L")
    mask_path = os.path.join(mask_folder, mask_fileName)
    mask_pil.save(mask_path)