import cv2
import os
from glob import glob
import natsort
import re

# Set your folder path
image_folder = "/home/ellina/Thesis/4DGaussians_Thesis/output_AVS_2/hypernerf/aleks"
output_video = "timestamp_dist_video_diverse.mp4"
fps = 5

# Get list of matching PNGs
images = glob(os.path.join(image_folder, "timestamp_dist_iter_*.png"))
images = natsort.natsorted(images)

# Separate images into multiples of 5 and others
pattern = re.compile(r"timestamp_dist_iter_(\d+).png")
multiples_of_5 = []
others = []

for img in images:
    match = pattern.search(os.path.basename(img))
    if match:
        iter_num = int(match.group(1))
        if iter_num % 50 == 0 and iter_num % 237 != 0:
            multiples_of_5.append(img)
        else:
            others.append(img)

# Final sequence
final_images = natsort.natsorted(multiples_of_5) + natsort.natsorted(others)

if not final_images:
    raise ValueError("No matching images found.")

# Get frame size from first image
frame = cv2.imread(final_images[0])
height, width, _ = frame.shape

# Write video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

for img_path in final_images:
    img = cv2.imread(img_path)
    video.write(img)

video.release()
print(f"Video saved to {output_video}")
