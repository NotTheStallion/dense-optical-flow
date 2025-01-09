import cv2 as cv
import numpy as np
import os



def farneback_optical_flow(frame1, frame2, prev_flow=None):
    """Compute dense optical flow using Farneback method."""
    gray1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
    flow = cv.calcOpticalFlowFarneback(
        gray1, gray2, prev_flow, pyr_scale=0.5, levels=5, winsize=50, iterations=3, poly_n=7, poly_sigma=1.5, flags=0
    )
    return flow

def pcaflow_optical_flow(frame1, frame2, prev_flow=None):
    """Compute dense optical flow using PCAFlow."""
    gray1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
    pcaflow = cv.optflow.createOptFlow_PCAFlow()
    flow = pcaflow.calc(gray1, gray2, prev_flow)
    return flow

# @critical : error display
def simpleflow_optical_flow(frame1, frame2, prev_flow=None):
    """Compute dense optical flow using SimpleFlow."""
    gray1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
    simpleflow = cv.optflow.createOptFlow_SimpleFlow()
    flow = simpleflow.calc(gray1, gray2, prev_flow)
    return flow

def deepflow_optical_flow(frame1, frame2, prev_flow=None):
    """Compute dense optical flow using DeepFlow."""
    gray1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
    deepflow = cv.optflow.createOptFlow_DeepFlow()
    flow = deepflow.calc(gray1, gray2, prev_flow)
    return flow

# @critical : not working
def dis_optical_flow(frame1, frame2, prev_flow=None):
    """Compute dense optical flow using DIS (Dense Inverse Search)."""
    gray1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
    dis = cv.optflow.createOptFlow_DIS(cv.optflow.DISOPTICAL_FLOW_PRESET_FAST)
    flow = dis.calc(gray1, gray2, prev_flow)
    return flow


def read_frames(directory):
    frames = []
    for filename in sorted(os.listdir(directory)):
        if filename.endswith((".jpg", ".png")):
            img = cv.imread(os.path.join(directory, filename))
            if img is not None:
                frames.append(img)
    return frames


def read_flow(filename):
        flow = cv.readOpticalFlow(filename)
        if flow is None:
            return None
        flow = flow[..., :2]
        return flow


def read_flows(directory):
    flows = []
    for filename in sorted(os.listdir(directory)):
        if filename.endswith(".flo"):
            flow = read_flow(os.path.join(directory, filename))
            if flow is not None:
                flows.append(flow)
    return flows


def write_flow(flow, filename):
    cv.writeOpticalFlow(filename, flow)


def flow2bgr(flow):
    h, w = flow.shape[:2]
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[..., 1] = 255

    mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
    bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

    # cv.imshow('Optical Flow', bgr)
    return bgr


def compute_ae(flow1, flow2):
    return np.mean(np.abs(flow1 - flow2))


def compute_epe(flow1, flow2):
    return np.mean(np.linalg.norm(flow1 - flow2, axis=-1))


def compute_mse(frame1, frame2):
    return np.mean((frame1 - frame2) ** 2)


def project(frame, flow):
    h, w = frame.shape[:2]
    flow_map = np.zeros_like(frame)

    for y in range(h):
        for x in range(w):
            dx, dy = flow[y, x]
            new_x = int(x + dx)
            new_y = int(y + dy)

            if 0 <= new_x < w and 0 <= new_y < h:
                flow_map[new_y, new_x] = frame[y, x]

    return flow_map


# Example usage
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    frame1 = cv.imread("MPI-Sintel_selection/training/clean/temple_3/frame_0002.png")
    frame2 = cv.imread("MPI-Sintel_selection/training/clean/temple_3/frame_0003.png")
    frame3 = cv.imread("MPI-Sintel_selection/training/clean/temple_3/frame_0004.png")
    
    flow_farneback = farneback_optical_flow(frame1, frame2)
    flow_pcaflow = pcaflow_optical_flow(frame1, frame2)
    # flow_simpleflow = simpleflow_optical_flow(frame1, frame2)
    flow_deepflow = deepflow_optical_flow(frame1, frame2)
    # flow_dis = dis_optical_flow(frame1, frame2)
    
    # display predicted next frame
    frame2_farneback = project(frame1, flow_deepflow)
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.title("Frame 1")
    plt.imshow(cv.cvtColor(frame1, cv.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.title("Frame 2")
    plt.imshow(cv.cvtColor(frame2, cv.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.title("Projected Frame 2 (Farneback)")
    plt.imshow(cv.cvtColor(frame2_farneback, cv.COLOR_BGR2RGB))
    plt.axis('off')
    
    plt.show()
    
    # numerical difference between flows
    for flow in [flow_farneback, flow_pcaflow, flow_deepflow]:
        print(flow.shape, np.mean(flow[..., 0]), np.mean(flow[..., 1]))

    cv.waitKey(0)
    cv.destroyAllWindows()
