import cv2
import numpy as np

def process_frame(frame):
    """
    Performs lane detection on a single video frame.

    Args:
        frame (numpy.ndarray): The video frame as a NumPy array.

    Returns:
        numpy.ndarray: The frame with detected lane lines overlaid (or None if no lines found).
    """
    
    # Preprocessing steps (adjust parameters as needed)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    blur = cv2.GaussianBlur(gray, (5, 5), 0)  # Apply Gaussian blur for noise reduction
    canny = cv2.Canny(blur, 50, 150)  # Canny edge detection to highlight lane markings

    # Region of interest (ROI) selection
    height, width = frame.shape[:2]
    roi_vertices = np.array([[(0, height // 2 + 60), (width, height // 2 + 60), (width, height), (0, height)]], dtype=np.int32)
    mask = np.zeros_like(canny)
    cv2.fillPoly(mask, roi_vertices, 255)  # Create a mask to focus on the road area
    masked_image = cv2.bitwise_and(canny, canny, mask=mask)  # Apply the mask

    # Hough transform for line detection
    lines = cv2.HoughLinesP(masked_image, rho=1, theta=np.pi/180, threshold=100, minLineLength=40, maxLineGap=50)

    # Initialize variables to store lane line coordinates
    left_line = None
    right_line = None

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]

            # Check for invalid slopes (avoid division by zero)
            if abs(x2 - x1) < 1e-5:
                continue  # Skip lines with near-zero horizontal distance

            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1

            # Classify lines based on slope (adjust thresholds if necessary)
            if -0.8 < slope < -0.4:  # Left lane
                left_line = (slope, intercept)
            elif 0.4 < slope < 0.8:  # Right lane
                right_line = (slope, intercept)

    # Draw detected lane lines (modify color and thickness as desired)
    if left_line is not None:
        slope, intercept = left_line
        y1 = height
        y2 = int(y1 * 3 // 5)  # Adjust y-coordinates to draw line on the road
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
        cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)  # Draw green line for left lane

    if right_line is not None:
        slope, intercept = right_line
        y1 = height
        y2 = int(y1 * 3 // 5)
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
        cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), 3)  # Draw red line for right lane

    return frame  # Return the frame with detected lines

# Video path (replace with your video file path)
video_path = "C:\\Users\\4a Freeboard\\Videos\\Mount_Road.mp4"

# Create video capture object
cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print("Error opening video stream or file")
    exit()

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Create a VideoWriter object to save the output video
output_path = "C:\\Users\\4a Freeboard\\Videos\\lane_detection_output.mp4"
out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

# Process each frame in the video
while cap.isOpened():
    # Capture frame-by-frame
    ret, frame = cap.read()

    # If frame is read correctly, process it
    if ret:
        # Process the frame
        processed_frame = process_frame(frame)

        # Display the frame with lane detection
        cv2.imshow('Lane Detection', processed_frame)

        # Write the frame to the output video file
        out.write(processed_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()


