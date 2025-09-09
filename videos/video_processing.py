import cv2
import os
def extract_frames_from_video(video_path, output_dir="frames"):
    """
    Extracts frames from a given mp4 file and saves them as images in the specified directory.

    :param video_path: Path to the input video file.
    :param output_dir: Directory where extracted frames will be saved.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Open the video file
    video_capture = cv2.VideoCapture(video_path)
    frame_count = 0

    while True:
        # Read a frame
        success, frame = video_capture.read()
        if not success:
            break

        # Save the frame as an image
        frame_filename = os.path.join(output_dir, f"frame_{frame_count:04d}.png")
        cv2.imwrite(frame_filename, frame)
        frame_count += 1

    video_capture.release()
    print(f"Extracted {frame_count} frames to {output_dir}")

extract_frames_from_video("a-billion_2.mp4")