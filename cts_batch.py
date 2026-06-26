import onnxruntime as ort
import numpy as np
import cv2
import argparse
import random
import time

def preprocess_image(image, input_size):
    image_resized = cv2.resize(image, input_size)
    image_normalized = image_resized.astype(np.float32) / 255.0
    image_normalized = image_normalized[:, :, ::-1]
    image_input = np.transpose(image_normalized, (2, 0, 1))
    image_input = np.expand_dims(image_input, axis=0)
    return image_input

def get_count_and_time(session, input_name, output_name, image, conf_threshold):
    input_size = (640, 640)

    start = time.perf_counter()

    image_input = preprocess_image(image, input_size)
    output = session.run([output_name], {input_name: image_input})[0][0]

    count = 0
    kept = []
    for detection in output:
        conf = float(detection[4])
        if conf > conf_threshold:
            count += 1
            kept.append(detection)

    # Simulated downstream workload: compare every kept detection with every other one
    dummy = 0.0
    for i in range(len(kept)):
        for j in range(len(kept)):
            dummy += float(kept[i][4]) * float(kept[j][4])

    end = time.perf_counter()

    return count, (end - start) * 1000

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

def attack_frame(session, input_name, output_name, frame, threshold, iters, strength):
    best_img = frame.copy()
    best_count, _ = get_count_and_time(session, input_name, output_name, best_img, threshold)

    for _ in range(iters):
        candidate = add_random_patch(best_img, strength)
        count, _ = get_count_and_time(session, input_name, output_name, candidate, threshold)

        if count > best_count:
            best_img = candidate
            best_count = count

    return best_img, best_count

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolov10n.onnx")
    parser.add_argument("--source", default="data/videos/road.mp4")
    parser.add_argument("--tresh", type=float, default=0.35)
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--iters", type=int, default=300)
    parser.add_argument("--strength", type=int, default=30)
    args = parser.parse_args()

    session = ort.InferenceSession(args.model)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    cap = cv2.VideoCapture(args.source)

    orig_counts = []
    attack_counts = []
    orig_times = []
    attack_times = []

    frame_id = 0

    while frame_id < args.frames:
        ok, frame = cap.read()
        if not ok:
            break

        orig_count, orig_time = get_count_and_time(
            session, input_name, output_name, frame, args.tresh
        )

        attacked_frame, attack_count = attack_frame(
            session, input_name, output_name, frame, args.tresh, args.iters, args.strength
        )

        attack_count_check, attack_time = get_count_and_time(
            session, input_name, output_name, attacked_frame, args.tresh
        )

        orig_counts.append(orig_count)
        attack_counts.append(attack_count_check)
        orig_times.append(orig_time)
        attack_times.append(attack_time)

        print(
            f"Frame {frame_id}: original={orig_count}, attack={attack_count_check}, "
            f"increase={attack_count_check - orig_count}"
        )

        if frame_id == 0:
            cv2.imwrite("batch_original_frame0.jpg", frame)
            cv2.imwrite("batch_attack_frame0.jpg", attacked_frame)

        frame_id += 1

    cap.release()

    print("\n===== CTS Batch Results =====")
    print("Frames tested:", len(orig_counts))
    print("Original avg detections:", np.mean(orig_counts))
    print("Attack avg detections:", np.mean(attack_counts))
    print("Average detection increase:", np.mean(np.array(attack_counts) - np.array(orig_counts)))
    print("Original avg inference ms:", np.mean(orig_times))
    print("Attack avg inference ms:", np.mean(attack_times))
    print("Saved example images: batch_original_frame0.jpg and batch_attack_frame0.jpg")