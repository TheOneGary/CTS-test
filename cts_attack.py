import onnxruntime as ort
import numpy as np
import cv2
import argparse
import random

def preprocess_image(image, input_size):
    original_shape = image.shape[:2]
    image_resized = cv2.resize(image, input_size)
    image_normalized = image_resized.astype(np.float32) / 255.0
    image_normalized = image_normalized[:, :, ::-1]
    image_input = np.transpose(image_normalized, (2, 0, 1))
    image_input = np.expand_dims(image_input, axis=0)
    return image_input, original_shape

def get_scores(session, input_name, output_name, image, conf_threshold):
    input_size = (640, 640)
    image_input, original_shape = preprocess_image(image, input_size)
    output = session.run([output_name], {input_name: image_input})[0][0]

    scores = []
    for detection in output:
        conf = float(detection[4])
        if conf > conf_threshold:
            scores.append(conf)

    return scores

def load_first_frame(source):
    img_formats = ["jpg", "jpeg", "png", "webp"]
    if source.split(".")[-1].lower() in img_formats:
        img = cv2.imread(source)
        if img is None:
            raise FileNotFoundError(f"Could not read image: {source}")
        return img

    cap = cv2.VideoCapture(source)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise FileNotFoundError(f"Could not read video: {source}")
    return frame

def add_random_patch(image, strength):
    attacked = image.copy()
    h, w = attacked.shape[:2]

    patch_w = random.randint(20, 120)
    patch_h = random.randint(20, 120)
    x = random.randint(0, max(1, w - patch_w))
    y = random.randint(0, max(1, h - patch_h))

    noise = np.random.randint(-strength, strength + 1, (patch_h, patch_w, 3))
    patch = attacked[y:y+patch_h, x:x+patch_w].astype(np.int16)
    patch = np.clip(patch + noise, 0, 255).astype(np.uint8)

    attacked[y:y+patch_h, x:x+patch_w] = patch
    return attacked

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolov10n.onnx")
    parser.add_argument("--source", default="data/videos/road.mp4")
    parser.add_argument("--tresh", type=float, default=0.35)
    parser.add_argument("--iters", type=int, default=300)
    parser.add_argument("--strength", type=int, default=40)
    args = parser.parse_args()

    session = ort.InferenceSession(args.model)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    original = load_first_frame(args.source)

    best_img = original.copy()
    best_scores = get_scores(session, input_name, output_name, best_img, args.tresh)
    best_count = len(best_scores)

    print("Original detections:", best_count)
    print("Original scores:", [round(s, 3) for s in best_scores])

    for i in range(args.iters):
        candidate = add_random_patch(best_img, args.strength)
        scores = get_scores(session, input_name, output_name, candidate, args.tresh)
        count = len(scores)

        if count > best_count:
            best_img = candidate
            best_scores = scores
            best_count = count
            print(f"Iter {i}: new best detections =", best_count)
            print("Scores:", [round(s, 3) for s in best_scores])

    cv2.imwrite("cts_original.jpg", original)
    cv2.imwrite("cts_attack.jpg", best_img)

    print("Final best detections:", best_count)
    print("Saved: cts_original.jpg and cts_attack.jpg")