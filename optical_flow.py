import cv2 as cv
import numpy as np
import os
import wandb
from utils import *


# os.environ['WANDB_DISABLED'] = 'true'


def run(directory, compute_optflow=farneback_optical_flow, display=False):
    frames = read_frames(directory)
    print(f"Read {len(frames)} frames {directory}")
    
    # get the last directory name
    category = directory.split('/')[-1]
    
    flow_directory = os.path.join(os.path.dirname(os.path.dirname(directory)), 'flow', category)
    gt_flows = read_flows(flow_directory)
    print(f"Read {len(gt_flows)} flows {flow_directory}")
    
    
    wandb_run = wandb.init(project="video_analysis", name=compute_optflow.__name__)
    wandb.config.update({"dir": category, "method": compute_optflow.__name__})   
    wandb.define_metric("frame")
    wandb.define_metric("*", step_metric="frame")

    flow_st = None
    
    for i in range(len(frames) - 1):
        frame1 = frames[i]
        frame2 = frames[i + 1]

        flow1 = compute_optflow(frame1, frame2, flow_st)
        # to get flow_directory move back to the parent of the parent directory and then to the flow directory and then to the category directory
        
        
        gt_flow = compute_optflow(frame1, frame2, flow_st)
            
        e_ae = compute_ae(flow1, gt_flow)
        e_epe = compute_epe(flow1, gt_flow)
        compensated_frame = project(frame1, flow1)
        e_mse = compute_mse(compensated_frame, frame2)
        flow_st = flow1

        wandb.log({"frame": (i/len(frames))*100, "AE": e_ae, "EPE": e_epe, "MSE": e_mse})

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
    directory = 'MPI-Sintel_selection/training/clean/temple_3'
    display = True
    run(directory, compute_optflow=pcaflow_optical_flow, display=display)
    run(directory, compute_optflow=deepflow_optical_flow, display=display)
    run(directory, compute_optflow=farneback_optical_flow, display=display)