import onnxruntime as ort
import numpy as np
import cv2
import argparse
import random

# Function to preprocess the image (resize and normalize)
def preprocess_image(image, input_size):    
    original_shape = image.shape[:2]  # Save original image size for later scaling   
    
    # Resize the image to the size expected by the model
    image_resized = cv2.resize(image, input_size)
    
    # Normalize (e.g., if the model requires input normalized to [0, 1])
    image_normalized = image_resized.astype(np.float32) / 255.0
    
    # Change the channel order from (B, G, R) to (R, G, B)
    image_normalized = image_normalized[:, :, ::-1]
    
    # Change image dimensions to (1, 3, height, width) as required by the ONNX model
    image_input = np.transpose(image_normalized, (2, 0, 1))
    image_input = np.expand_dims(image_input, axis=0)
    
    return image_input, original_shape

# Function to perform post-processing on the detection results
def postprocess_output(output, original_shape, input_size, conf_threshold=0.5):
    boxes, scores, class_ids = [], [], []
    
    for detection in output:        
        confidence = detection[4]  # Confidence score
        if confidence > conf_threshold:  # Filter out detections below threshold
            x1, y1, x2, y2 = detection[0:4]  # Bounding box coordinates
            class_id = int(detection[5])  # Class ID

            # Scale bounding box coordinates back to the original size
            x1 = x1 / input_size[0]
            y1 = y1 / input_size[1]
            x2 = x2 / input_size[0]
            y2 = y2 / input_size[1]

            x1 = int(x1 * original_shape[1])
            y1 = int(y1 * original_shape[0])
            x2 = int(x2 * original_shape[1])
            y2 = int(y2 * original_shape[0])
            
            # Append the bounding box, confidence score, and class ID to respective lists
            boxes.append([x1, y1, x2, y2])
            scores.append(float(confidence))
            class_ids.append(class_id)                         
    
    return boxes, scores, class_ids

# Function to load source (either image or video)
def loadSource(source_file):
    img_formats = ['jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng', 'webp', 'mpo']  # Supported image formats
    key = 1 # Default key for video mode, which waits for 1ms per frame
    frame = None
    cap = None

    # If the source is "0", open webcam
    if(source_file == "0"):
        image_type = False
        source_file = 0    
    else:
        image_type = source_file.split('.')[-1].lower() in img_formats  # Determine if the source is an image

    # If it's an image, read the file; otherwise, open video
    if(image_type):
        frame = cv2.imread(source_file)
        key = 0  # For image, the key should wait indefinitely
    else:
        cap = cv2.VideoCapture(source_file)

    return image_type, key, frame, cap

if __name__ == '__main__':
    # Add argument parser to get command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, default="data/videos/road.mp4", help="Video")
    parser.add_argument("--names", type=str, default="data/class.names", help="Object Names")
    parser.add_argument("--model", type=str, default="yolov10n.onnx", help="Pretrained Model")
    parser.add_argument("--tresh", type=float, default=0.35, help="Confidence Threshold")    
    args = parser.parse_args()

    # Load the ONNX model
    model_path = args.model
    session = ort.InferenceSession(model_path)

    # Get the input and output layers from the model
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # Define the input size for the model
    input_size = (640, 640)  

    # Define confidence threshold for detection
    conf_thres = args.tresh

    # Load class names from the .names file and assign random colors to each class
    class_names = []
    with open(args.names, "r") as f:
        class_names = [cname.strip() for cname in f.readlines()]
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in class_names]
    
    # Load source (image or video)
    source_file = args.source 
    image_type, key, frame, cap = loadSource(source_file)
    grabbed = True  # Flag to check if frame is successfully captured

    while(1):
        if not image_type:  # If it's a video, capture each frame
            (grabbed, frame) = cap.read()

        if not grabbed:  # If no frame is grabbed, exit the loop
            exit()
        
        image = frame.copy()  # Make a copy of the frame for processing

        # Preprocess the image for the model
        image_input, original_shape = preprocess_image(image, input_size)

        # Perform inference using the ONNX model
        output = session.run([output_name], {input_name: image_input})[0]

        # Post-process the output from the model
        boxes, scores, class_ids = postprocess_output(output[0], original_shape, input_size, conf_thres)
 
        # Gary added
        print("Threshold:", conf_thres)
        print("Detections above threshold:", len(boxes))
        print("Scores:", [round(s, 3) for s in scores])

        # Draw the detection results (bounding boxes and class names)
        for box, score, class_id in zip(boxes, scores, class_ids):
            x1, y1, x2, y2 = box

            # Class name with confidence score
            class_name = class_names[class_id]    
            score = round(float(score), 3)            
            class_name += f' {str(score)}'
            
            # Draw bounding box and class name on the image
            cv2.rectangle(image, (x1, y1), (x2, y2), colors[class_id], 2)
            cv2.putText(image, class_name, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, colors[class_id], 2)

        grabbed = False  # Reset the grabbed flag for the next loop iteration
        cv2.imshow("Object Detection", image)  # Show the result
        if cv2.waitKey(key) ==  ord('q'):  # If 'q' is pressed, exit the loop
            break

    cv2.destroyAllWindows()  # Close all OpenCV windows
