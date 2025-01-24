import cv2
import os
def save_frame(frame, folder, filename):
    """
    Save a frame to specified folder
    
    params:
        frame: The image frame to be saved
        folder: The folder path where the image frame will be saved
        filename: The filename for the image frame
    """
    if not os.path.exists(folder):
        os.makedirs(folder)
    cv2.imwrite(os.path.join(folder, filename), frame)
    
def process_video(video_path, output_folder):
    '''

    params:
        video_path: The folder path to the input video file
        output_folder: The folder path where extracted frames will be saved
    Returns:
        fps: The frame rate of the video

    '''
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Error opening video file: {video_path}")
    
    #Get the frame rate of the video
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_number = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        save_frame(frame, output_folder, f"frame_{frame_number:04d}.png")
        frame_number += 1

    cap.release()

    return fps


    