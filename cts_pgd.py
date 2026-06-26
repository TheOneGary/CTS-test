import torch
import cv2
import numpy as np
import argparse
from ultralytics import YOLO

def load_first_frame(source):
    cap = cv2.VideoCapture(source)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        frame = cv2.imread(source)
    if frame is None:
        raise FileNotFoundError(source)
    return frame

def preprocess(frame):
    img = cv2.resize(frame, (640, 640))
    img = img[:, :, ::-1] / 255.0
    x = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0)
    return x

def collect_tensors(obj):
    tensors = []
    if torch.is_tensor(obj):
        tensors.append(obj)
    elif isinstance(obj, (list, tuple)):
        for item in obj:
            tensors += collect_tensors(item)
    elif isinstance(obj, dict):
        for item in obj.values():
            tensors += collect_tensors(item)
    return tensors

def cts_loss(model_output, threshold):
    tensors = collect_tensors(model_output)

    candidates = []
    for t in tensors:
        if t.requires_grad and t.ndim == 3:
            candidates.append(t)

    if not candidates:
        raise RuntimeError("No gradient-connected prediction tensor found.")

    pred = max(candidates, key=lambda x: x.numel())

    if pred.shape[1] < pred.shape[2]:
        pred = pred.permute(0, 2, 1)

    # YOLO raw class/objectness scores
    scores = pred[..., 4:]

    # Do NOT sigmoid twice if values already look like probabilities
    if scores.max() > 1 or scores.min() < 0:
        scores = scores.sigmoid()

    conf = scores.max(dim=-1).values

    # CTS target: push many predictions just ABOVE threshold
    above = torch.sigmoid(80 * (conf - threshold)).sum()

    # Also avoid making only 1-2 boxes extremely confident
    near_threshold = torch.exp(-((conf - (threshold + 0.05)) ** 2) / 0.01).sum()

    loss = -(above + 0.5 * near_threshold)

    return loss

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolov10n.pt")
    parser.add_argument("--source", default="data/videos/road.mp4")
    parser.add_argument("--tresh", type=float, default=0.50)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--eps", type=float, default=8/255)
    parser.add_argument("--alpha", type=float, default=1/255)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    yolo = YOLO(args.model)
    model = yolo.model.to(device)

    # Important: train mode gives raw gradient-connected outputs
    model.train()

    frame = load_first_frame(args.source)
    x0 = preprocess(frame).to(device)

    delta = torch.zeros_like(x0, requires_grad=True)

    for step in range(args.steps):
        adv = torch.clamp(x0 + delta, 0, 1)

        output = model(adv)
        loss = cts_loss(output, args.tresh)

        loss.backward()

        with torch.no_grad():
            delta -= args.alpha * delta.grad.sign()
            delta.clamp_(-args.eps, args.eps)
            delta.grad.zero_()

        print(f"step {step+1}/{args.steps}, loss={loss.item():.3f}")

    adv = torch.clamp(x0 + delta, 0, 1)

    orig_img = (x0[0].permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)
    adv_img = (adv[0].permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8)

    cv2.imwrite("cts_pgd_original.jpg", orig_img[:, :, ::-1])
    cv2.imwrite("cts_pgd_attack.jpg", adv_img[:, :, ::-1])

    print("Saved cts_pgd_original.jpg")
    print("Saved cts_pgd_attack.jpg")