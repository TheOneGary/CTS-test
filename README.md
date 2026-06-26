# CTS-test
(For differences in detection): python cts_batch.py --model yolov10n.onnx --source data\videos\road.mp4 --tresh 0.35 --frames 30 --iters 1000 --strength 30



========================================================================================================================================================


(To get the video with boxes): 

First: python cts_batch.py --model yolov10n.onnx --source data\videos\road.mp4 --tresh 0.35 --frames 30 --iters 1000 --strength 30

Second: python detection.py --model yolov10n.onnx --source cts_attack_video.mp4 --tresh 0.35
