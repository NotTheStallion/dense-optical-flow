import cv2 as cv
import numpy as np
import os
import wandb


# os.environ['WANDB_DISABLED'] = 'true'


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


# flow1 = compute_optical_flow(frame1, frame2)
# e = compute_ae(flow-st, flow1) # compute_spe(flow-st, flow1)
# componsated_frame = project(frame1, flow1) # done using remap in opencv
# e = compute_mse(componsated_frame, frame2)

def compute_optical_flow(frame1, frame2):
    gray1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
    flow = cv.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return flow

def compute_optical_flow_lk(frame1, frame2):
    gray1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
    p0 = cv.goodFeaturesToTrack(gray1, mask=None, **feature_params)
    p1, _, _ = cv.calcOpticalFlowPyrLK(gray1, gray2, p0, None, **lk_params)
    flow = np.zeros((frame1.shape[0], frame1.shape[1], 2), dtype=np.float32)
    for new, old in zip(p1, p0):
        a, b = new.ravel()
        c, d = old.ravel()
        flow[int(d), int(c)] = [a - c, b - d]
    return flow

def compute_optical_flow_tvl1(frame1, frame2):
    gray1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
    tvl1 = cv.optflow.DualTVL1OpticalFlow_create()
    flow = tvl1.calc(gray1, gray2, None)
    return flow

def compute_optical_flow_deepflow(frame1, frame2):
    gray1 = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(frame2, cv.COLOR_BGR2GRAY)
    deepflow = cv.optflow.createOptFlow_DeepFlow()
    flow = deepflow.calc(gray1, gray2, None)
    return flow

feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))

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


def run(directory, method='farneback', display=False):
    frames = read_frames(directory)
    print(f"Read {len(frames)} frames")
    
    # get upper upper directory name
    upper_upper_directory = os.path.basename(os.path.dirname(os.path.dirname(directory)))
    print(f"Upper upper directory: {upper_upper_directory}")
    
    wandb_run = wandb.init(project="video_analysis", name=method)
    wandb.config.update({"upper_upper_directory": upper_upper_directory, "method": method})
    wandb.define_metric("frame")
    wandb.define_metric("*", step_metric="frame")

    flow_st = compute_optical_flow_lk(frames[0], frames[1])
    for i in range(len(frames) - 1):
        frame1 = frames[i]
        frame2 = frames[i + 1]
        if method == 'farneback':
            flow1 = compute_optical_flow(frame1, frame2)
        elif method == 'lk':
            flow1 = compute_optical_flow_lk(frame1, frame2)
        elif method == 'tvl1':
            flow1 = compute_optical_flow_tvl1(frame1, frame2)
        elif method == 'deepflow':
            flow1 = compute_optical_flow_deepflow(frame1, frame2)
        e_ae = compute_ae(flow_st, flow1)
        e_epe = compute_epe(flow_st, flow1)
        compensated_frame = project(frame1, flow1)
        e_mse = compute_mse(compensated_frame, frame2)
        flow_st = flow1
        # print(i, " ",len(frames), " ", (i/len(frames))*100)
        wandb.log({"frame": (i/len(frames))*100, "AE": e_ae, "EPE": e_epe, "MSE": e_mse})
        # print(f"Frame {i} to {i+1} - AE: {e_ae} - EPE: {e_epe}, MSE: {e_mse}")
        # Display the optical flow
        if display:
            hsv = np.zeros_like(frame1)
            hsv[..., 1] = 255

            mag, ang = cv.cartToPolar(flow1[..., 0], flow1[..., 1])
            hsv[..., 0] = ang * 180 / np.pi / 2
            hsv[..., 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
            bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)

            cv.imshow('Optical Flow', bgr)
            if cv.waitKey(1) & 0xFF == ord('q'):
                break

    wandb_run.finish()

    if display:
        cv.destroyAllWindows()


if __name__ == '__main__':
    directory = 'GITW_selection/CanOfCocaCola/CanOfCocaColaPlace3Subject1/Frames'
    display = False
    run(directory, method='lk', display=display)
    run(directory, method='farneback', display=display)
    
    directory = 'GITW_selection/Bowl/BowlPlace1Subject2/Frames'
    run(directory, method='lk', display=display)
    run(directory, method='farneback', display=display)
    
    directory = 'GITW_selection/Rice/RicePlace6Subject3/Frames'
    run(directory, method='lk', display=display)
    run(directory, method='farneback', display=display)