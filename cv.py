import cv2
import numpy as np

video_path = '/content/sample_data/WHATSAAP ASSIGNMENT.mp4'
cap = cv2.VideoCapture(video_path)


def optical_flow(prev_frame, curr_frame):
    prev_frame_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_frame_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    prev_frame_gray_np = np.ascontiguousarray(prev_frame_gray)
    curr_frame_gray_np = np.ascontiguousarray(curr_frame_gray)
    flow = cv2.calcOpticalFlowFarneback(prev_frame_gray_np, curr_frame_gray_np, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return flow

def average_speed(flow):
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    avg_speed = np.mean(mag)
    return avg_speed   

prev_frame = None
num_dribbles = 0
total_speed = 0

while cap.isOpened():
    ret, curr_frame = cap.read()
    if not ret:
        break

    if prev_frame is not None:
        flow = optical_flow(prev_frame, curr_frame)
        speed = average_speed(flow)
        total_speed += speed

        # Threshold to determine if a dribble is being performed
        if speed > 10:
            num_dribbles += 1

    prev_frame = curr_frame     



avg_speed = total_speed / num_dribbles
print(f'Number of dribbles: {num_dribbles}')
print(f'Average speed of dribbles: {avg_speed}')
