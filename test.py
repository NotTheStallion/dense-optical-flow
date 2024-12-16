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

# params for ShiTomasi corner detection
feature_params = dict(maxCorners=200,  # Increased the number of corners to detect
                      qualityLevel=0.2,  # Lowered the quality level to detect more corners
                      minDistance=5,  # Reduced the minimum distance between corners
                      blockSize=5)  # Reduced the block size for corner detection

# Parameters for lucas kanade optical flow
lk_params = dict(winSize=(21, 21),  # Increased the window size for better tracking
                 maxLevel=3,  # Increased the number of pyramid levels
                 criteria=(cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 15, 0.01))  # Modified criteria for termination

# Create some random colors
color = np.random.randint(0, 255, (200, 3))  # Increased the number of colors to match the increased number of corners

# Take first frame and find corners in it
old_frame = frames[0]
old_gray = cv.cvtColor(old_frame, cv.COLOR_BGR2GRAY)
p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

# Create a mask image for drawing purposes
mask = np.zeros_like(old_frame)
for i, frame in enumerate(frames[1:]):
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # calculate optical flow
    p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

    # Select good points
    if p1 is not None and st is not None:
        good_new = p1[st == 1]
        good_old = p0[st == 1]
    else:
        good_new = np.array([])
        good_old = np.array([])

    # If the number of good points falls below a threshold, re-detect corners
    if len(good_new) < 10:
        p0 = cv.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
        good_new = p0
        good_old = p0

    # draw the tracks
    for j, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        mask = cv.line(mask, (int(a), int(b)), (int(c), int(d)), color[j].tolist(), 2)
        frame = cv.circle(frame, (int(a), int(b)), 5, color[j].tolist(), -1)
    img = cv.add(frame, mask)

    # Create the output directory if it doesn't exist
    output_directory = os.path.join(directory, '../opt_flow')
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Save the image
    output_filename = os.path.join(output_directory, f'opt_flow_{i:05d}.jpg')
    cv.imwrite(output_filename, img)

    # Now update the previous frame and previous points
    old_gray = frame_gray.copy()
    p0 = good_new.reshape(-1, 1, 2)