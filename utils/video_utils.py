import cv2
from PIL import Image

def extract_frames(video_path, fps_interval=1):
    """
    Extract frames from video for moderation.
    :param video_path: path to video file
    :param fps_interval: how many frames per second to sample
    :return: list of PIL.Image frames
    """
    frames = []
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 1
    interval = max(int(fps * fps_interval), 1)
    
    frame_idx = 0
    success = True
    while success:
        success, frame = cap.read()
        if not success:
            break
        if frame_idx % interval == 0:
            # Convert to RGB PIL Image
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        frame_idx += 1

    cap.release()
    return frames
