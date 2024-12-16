import cv2 as cv
import numpy as np
import os


# Reading frames from directory 
def read_frames(directory):
    frames = []
    for filename in sorted(os.listdir(directory)):
        if filename.endswith(".jpg"):
            img = cv.imread(os.path.join(directory, filename))
            if img is not None:
                frames.append(img)
    return frames

# directory = 'GITW_selection/Bowl/BowlPlace1Subject2/Frames'
directory = 'GITW_selection/CanOfCocaCola/CanOfCocaColaPlace3Subject1/Frames'
frames = read_frames(directory)
print(f"Read {len(frames)} frames")

# flow1 = compute_optical_flow(frame1, frame2)
# e = compute_ae(flow-st, flow1) # compute_spe(flow-st, flow1)
# componsated_frame = project(frame1, flow1) # done using remap in opencv
# e = compute_mse(componsated_frame, frame2)

def compute_optical_flow(frame1, frame2):
    gray1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
    flow = cv.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return flow

def compute_ae(flow1, flow2):
    return np.mean(np.abs(flow1 - flow2))

def compute_epe(flow1, flow2):
    return np.mean(np.linalg.norm(flow1 - flow2, axis=-1))

def project(frame, flow):
    h, w = flow.shape[:2]
    flow_map = np.column_stack((flow[..., 0].flatten(), flow[..., 1].flatten()))
    remap = np.column_stack((np.repeat(np.arange(w), h), np.tile(np.arange(h), w)))
    remap = remap + flow_map
    remap = remap.reshape(h, w, 2).astype(np.float32)
    return cv.remap(frame, remap, None, cv.INTER_LINEAR)

def compute_mse(frame1, frame2):
    return np.mean((frame1 - frame2) ** 2)

flow_st = compute_optical_flow(frames[0], frames[1])
for i in range(len(frames) - 1):
    frame1 = frames[i]
    frame2 = frames[i + 1]
    flow1 = compute_optical_flow(frame1, frame2)
    e_ae = compute_ae(flow_st, flow1)
    e_epe = compute_epe(flow_st, flow1)
    compensated_frame = project(frame1, flow1)
    e_mse = compute_mse(compensated_frame, frame2)
    flow_st = flow1
    print(f"Frame {i} to {i+1} - AE: {e_ae} - EPE: {e_epe}, MSE: {e_mse}")
    # Display the optical flow
    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255

    mag, ang = cv.cartToPolar(flow1[..., 0], flow1[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

    cv.imshow('Optical Flow', bgr)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cv.destroyAllWindows()