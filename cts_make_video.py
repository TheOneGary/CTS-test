import cv2
import subprocess
import os

# This runs detection.py on the original video.
# For attacked frames, you need cts_batch.py to save attacked frames first.

print("To view original video with boxes:")
print("python detection.py --model yolov10n.onnx --source data\\videos\\road.mp4 --tresh 0.35")

print("\nTo view one attacked frame with boxes:")
print("python detection.py --model yolov10n.onnx --source batch_attack_frame0.jpg --tresh 0.35")