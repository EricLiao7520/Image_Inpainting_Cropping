import os
import argparse
import tempfile
from segment import generate_masks
from cropping import crop_frames
from crop_frame import process_video
from util import inpaint_frames, ocr_padded, letterBox_detection, extract_audio, combine_frames_and_audio
def create_ocr_for_frames(frames_folder, processed_dir, crop):
    '''
    OCR detection and padding for the frames.
    '''
    letterBox_detection(frames_folder, 0.2)
    frame_list, frame_file_list, ocr_list = ocr_padded(frames_folder, processed_dir, crop)
    return frame_list, frame_file_list, ocr_list

def process_video_with_steps(video_path, output_dir, crop, steps):
    '''
    Main processing pipeline for subtitle removal.
    '''
    processed_dir = os.path.join(output_dir, "processed_frames")
    final_video = os.path.join(output_dir, "final_inpainted_subtitles_removed.mp4")
    os.makedirs(processed_dir, exist_ok=True)

    with tempfile.TemporaryDirectory() as frames_folder, tempfile.TemporaryDirectory() as audio_path:

        mask_folder = frames_folder
        #Extract audio files from input video
        extract_audio(video_path, audio_path)

        # frame preprocessing
        frame_rate = process_video(video_path, frames_folder)

        # Create OCR bounding boxes for subtitle
        frame_list, frame_file_list, ocr_list = create_ocr_for_frames(frames_folder, processed_dir, crop)

        # Create masks for subtitle regions
        generate_masks(mask_folder, frame_list, frame_file_list, ocr_list, crop)

        if crop:
            # If crop is True, do cropping
            crop_frames(frames_folder, mask_folder, processed_dir)
        else:
            # If crop is False, do inpainting
            inpaint_frames(frames_folder, processed_dir, steps)
            
        # Merge frames and audio back to original video
        combine_frames_and_audio(processed_dir, audio_path, final_video, frame_rate)
def main():
    # Argument parsing
    parser = argparse.ArgumentParser(description="Subtitle Removal Pipeline")
    parser.add_argument(
        "--video_dir",
        type=str,
        required=True,
        help="Path to the input video file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to the output folder"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="Number of ddim sampling steps (default: 50)."
    )
    parser.add_argument(
        "--crop",
        action='store_true',
        help="If set to True, physically crop out the subtitle region"
    )
    args = parser.parse_args()
    crop = args.crop
    steps = args.steps
    video_path = args.video_dir
    output_dir = os.path.abspath(args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Input Video: {video_path}")
    print(f"Output Directory: {output_dir}")

    # Start processing
    process_video_with_steps(video_path, output_dir, crop, steps)

if __name__ == "__main__":
    main()