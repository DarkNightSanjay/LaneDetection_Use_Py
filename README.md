# Lane Detection Video Processing

This project provides a Python script that processes a video file to detect lane markings on the road using OpenCV. The script detects lane lines in each video frame, overlays the detected lines, and saves the processed video with lane detection.

## Prerequisites

Before running the script, ensure you have the following installed:

- Python 3.x
- OpenCV library
- NumPy library

You can install the required Python packages using pip:


    pip install opencv-python numpy

## Script Description
    process_frame(frame)
This function processes each frame of the video to detect lane lines. The process involves:

 - Grayscale Conversion: Converts the frame to grayscale to simplify processing.
 - Gaussian Blur: Applies a Gaussian blur to reduce noise in the image.
 - Canny Edge Detection: Highlights the edges in the frame to make lane markings more prominent.
 - Region of Interest (ROI) Masking: Masks the image to focus on the road area where lane lines are typically located.
 - Hough Line Transform: Detects straight lines in the masked image.
 - Line Classification: Classifies detected lines into left and right lane lines based on slope.
 - Lane Line Drawing: Draws green and red lines on the detected left and right lanes, respectively.
## Video Processing
The script reads a video file, processes each frame using the process_frame function, and writes the processed frames to an output video.
The processed video displays the detected lane lines on the road.
How to Use
Place Your Input Video: Ensure the input video file is located at the specified path in the script (video_path variable).

Run the Script: Execute the script using Python. The script will read the input video, apply lane detection, and save the processed video to the specified output path.


    python lane_detection.py
## Example Paths
    Input Video Path: C:\\Users\\4a Freeboard\\Videos\\Mount_Road.mp4
    Output Video Path: C:\\Users\\4a Freeboard\\Videos\\lane_detection_output.mp4
## Notes
 - Ensure that the input video file exists and the specified paths are correct.
 - You can change the input and output paths as needed by modifying the video_path and output_path variables in the script.
 - The output video will be saved in the MP4 format using the mp4v codec.
