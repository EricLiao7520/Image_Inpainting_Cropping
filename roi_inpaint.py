import torch
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse, os, sys, glob
from omegaconf import OmegaConf
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from main import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
#from ldm.models.diffusion.plms import PLMSSampler

from image_inpainting.util import (
    find_bounding_boxes_from_mask,
    align_bbox_to_64,
    pad_to_64,
    unpad_to_original,
    merge_overlapping_boxes,
    crop_with_box,
    paste_inpainted_batch
)

def load_single_image_and_masks(image_path, mask_path):
    '''
    Load image and corresponding mask.
    '''
    image = np.array(Image.open(image_path).convert("RGB"), dtype=np.float32) / 255.0
    mask  = np.array(Image.open(mask_path).convert("L"), dtype=np.float32) / 255.0
    return image, mask

def load_images_and_masks(image_paths, mask_paths):
    '''
    Load multiple image and mask pairs in parallel.
    '''
    images_np, masks_np = [], []
    with ThreadPoolExecutor() as executor:
        results = list(tqdm(
            executor.map(load_single_image_and_masks, image_paths, mask_paths),
            total=len(image_paths),
            desc="Loading files"
        ))
    for image, mask in results:
        images_np.append(image)  # [H, W, 3] in [0,1]
        masks_np.append(mask)    # [H, W] in [0,1]
    return images_np, masks_np

def make_batch(images_np, masks_np, device):
    '''
    Create a batch of images and mask ready for inpainting
    '''
    images_tensor = torch.from_numpy(
        np.stack([img.transpose(2, 0, 1) for img in images_np], axis=0)
    ).float().to(device)
    
    masks_tensor = torch.from_numpy(
        np.stack([msk[None] for msk in masks_np], axis=0)
    ).float().to(device)
    
    # Binarize mask nad prepare images
    masks_tensor = (masks_tensor >= 0.5).float()
    masked_images_tensor = (1.0 - masks_tensor) * images_tensor

    # Scale [0,1] => [-1,1]
    images_tensor        = images_tensor        * 2.0 - 1.0
    masks_tensor         = masks_tensor         * 2.0 - 1.0
    masked_images_tensor = masked_images_tensor * 2.0 - 1.0
    
    return {
        "image": images_tensor,
        "mask": masks_tensor,
        "masked_image": masked_images_tensor
    }

def inpaint_batch(model, sampler, batch):
    '''
    Inpaint the images using the latent diffusion model.
    '''
    c = model.cond_stage_model.encode(batch["masked_image"])
    cc = torch.nn.functional.interpolate(batch["mask"], size=c.shape[-2:])
    c = torch.cat((c, cc), dim=1)

    shape = (c.shape[1] - 1,) + c.shape[2:]
    samples_ddim, _ = sampler.sample(
        S=global_opt.steps,
        conditioning=c,
        batch_size=c.shape[0],
        shape=shape,
        verbose=False,
    )
    x_samples_ddim = model.decode_first_stage(samples_ddim)
    
    #Convert to [0, 1] range
    image           = torch.clamp((batch["image"]       + 1.0)/2.0, 0.0, 1.0)
    mask            = torch.clamp((batch["mask"]        + 1.0)/2.0, 0.0, 1.0)
    predicted_image = torch.clamp((x_samples_ddim       + 1.0)/2.0, 0.0, 1.0)

    return (1 - mask)*image + mask*predicted_image

def process_inpainting_for_single_image(image_np, mask_np, model, sampler, device, outdir, autocast_context):
    '''
    Process a single image and its mask for inpainting
    '''
    padded_img, (pad_w, pad_h), (off_y, off_x) = pad_to_64(image_np)
    padded_mask, _, _ = pad_to_64(mask_np)

    # Prepare batch
    batch = make_batch([padded_img], [padded_mask], device=device)

    # Inpaint the batch
    with autocast_context():
        inpainted_batch = inpaint_batch(model, sampler, batch)

    # Convert to numpy and unpad the result
    inpainted_batch = (inpainted_batch.detach().cpu().numpy() * 255).astype(np.uint8)
    inpainted_batch = inpainted_batch[0].transpose(1, 2, 0)
    unpadded_batch = unpad_to_original(inpainted_batch, image_np.shape[:2], (off_y, off_x))

    # Merge the inpainted batch where is True
    final_image = (image_np * 255).astype(np.uint8)
    mask_uint8 = (mask_np * 255).astype(np.uint8)
    mask_bool = mask_uint8 > 0  # Assuming mask is binary
    final_image[mask_bool] = unpadded_batch[mask_bool]

    # Save the final image
    outpath = os.path.join(global_opt.outdir, os.path.basename(images[i]))
    Image.fromarray(final_image).save(outpath)
                    
def process_inpainting_for_batch(image_np, mask_np, model, sampler, device, outdir, autocast_context, batch_size = 1):
    '''
    Process images and masks in batch for inpainting
    '''
    bboxes = find_bounding_boxes_from_mask(mask_np)
    bboxes = merge_overlapping_boxes(bboxes)
    cropped_images = []
    cropped_masks = []
    box_list = []
    H, W = image_np.shape[:2]

    for box in bboxes:
        x1, y1, w, h = box
        x2 = x1 + w - 1
        y2 = y1 + h - 1
        new_x1, new_y1, new_x2, new_y2 = align_bbox_to_64([x1,y1,x2,y2], W, H)
        expanded_box = [new_x1, new_y1, new_x2, new_y2]
        cimg, cmask = crop_with_box(image_np, mask_np, expanded_box)
        cimg_h, cimg_w = cimg.shape[:2]
        padded_cimg, (p_w, p_h), (off_y, off_x) = pad_to_64(cimg)
        padded_cmsk, _, _ = pad_to_64(cmask)
        cropped_images.append(padded_cimg)
        cropped_masks.append(padded_cmsk)
        box_list.append((expanded_box, cimg_h, cimg_w, off_y, off_x))

    inpainted_batches =[]
    for start_idx in range(0, len(cropped_images), batch_size):
        end_idx = start_idx + batch_size
        batch_images_np = cropped_images[start_idx:end_idx]
        batch_masks_np  = cropped_masks[start_idx:end_idx]
        
        # Make the batch
        batch = make_batch(batch_images_np, batch_masks_np, device=device)
        
        # mixed precision
        with autocast_context():
            inpainted_batch = inpaint_batch(model, sampler, batch)
        
        # Save results
        # inpainted_batch shape: [B, 3, H, W], in [0,1]
        inpainted_batch = (inpainted_batch.detach().cpu().numpy() * 255).astype(np.uint8)
        for bi in range(inpainted_batch.shape[0]):
            batch = inpainted_batch[bi].transpose(1, 2, 0)
            inpainted_batches.append(batch)
    #Reconstruct final image by pasting inpainting batches batches back to original images
    final_image = (image_np * 255).astype(np.uint8)
    for idx, (box, cimg_h, cimg_w, off_y, off_x) in enumerate(box_list):
        image_final =  unpad_to_original(
                inpainted_batches[idx],
                (cimg_h, cimg_w),
                (off_y, off_x)
            )
        final_image = paste_inpainted_batch(final_image, image_final, box)

    outpath = os.path.join(global_opt.outdir, os.path.basename(images[i]))
    Image.fromarray(final_image).save(outpath)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--indir",  type=str, required=True,
                        help="dir containing image-mask pairs (`example.png` and `example_mask.png`)")
    parser.add_argument("--outdir", type=str, required=True,
                        help="dir to write results to")
    parser.add_argument("--steps",  type=int, default=50,
                        help="number of DDIM sampling steps")
    parser.add_argument("--batch_size", type=int, default=1, #For V100(32g vram) 720p
                        help="batch size for inpainting")
    global_opt = parser.parse_args()

    # Gather file paths
    masks  = sorted(glob.glob(os.path.join(global_opt.indir, "*_mask.png")))
    images = [x.replace("_mask.png", ".png") for x in masks]
    print(f"Found {len(masks)} mask-image pairs.")

    # Load config & model
    try:
        config = OmegaConf.load("../inpaint/models/ldm/inpainting_big/config.yaml")
        model = instantiate_from_config(config.model)
        sd = torch.load("../inpaint/models/ldm/inpainting_big/last.ckpt", 
                        map_location="cuda", 
                        weights_only=False)
        model.load_state_dict(sd["state_dict"], strict=False)
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    sampler = DDIMSampler(model)
    #sampler = PLMSSampler(model)

    # Load all images/masks into CPU memory as numpy arrays only once
    all_images_np, all_masks_np = load_images_and_masks(images, masks)
    os.makedirs(global_opt.outdir, exist_ok=True)

    # autocast context for half precision if on CUDA
    autocast_context = lambda:torch.autocast(device_type='cuda', dtype=torch.float16) if device.type == 'cuda' else nullcontext

    # Process in batches
    if(all_images_np[0].shape[0] < 720 or all_images_np[0].shape[1] < 720):
    # Process Whole-Image Inpainting Group
        with torch.inference_mode():
            with model.ema_scope():
                for i in range(len(all_images_np)):
                    # Pad image and mask to multiples of 64
                    process_inpainting_for_single_image(all_images_np[i], all_masks_np[i], model, sampler, 
                                                        device, global_opt.outdir, autocast_context)
                    torch.cuda.empty_cache()
    else:
        with torch.inference_mode():
            with model.ema_scope():
                for i in range(len(all_images_np)):
                    process_inpainting_for_batch(all_images_np[i], all_masks_np[i], model, sampler, device, 
                                                global_opt.outdir, autocast_context, global_opt.batch_size)
                    torch.cuda.empty_cache()