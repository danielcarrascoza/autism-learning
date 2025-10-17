import os
import csv
import cv2
import numpy as np
from glob import glob

def load_dataset(label_csv, base_folder, engagement_slot=2, limit=1000):
    X, y = [], []

    with open(label_csv, "r") as f:
        reader = csv.reader(f)
        next(reader)  # skip header

        for row in reader:
            video_name = row[0].replace(".avi", "")
            try:
                raw_label = float(row[engagement_slot])
            except ValueError:
                continue

            # 🔍 Recursive search pattern (all levels)
            search_pattern = os.path.join(base_folder, "**", f"{video_name}.avi")
            matches = glob(search_pattern, recursive=True)

            if not matches:
                print(f"⚠️  Video not found for {video_name}")
                continue

            video_path = matches[0]
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"⚠️  Could not open video {video_path}")
                continue

            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frames.append(np.mean(gray))
            cap.release()

            if len(frames) == 0:
                continue

            avg_feature = np.mean(frames)
            features = [avg_feature, avg_feature / 2, avg_feature / 3]

            X.append(features)
            if raw_label == 0:
                y.append(0)
            if raw_label == 1:
                y.append(.175)
            if raw_label == 2:
                y.append(.4)
            if raw_label == 3:
                y.append(1)
            if len(X) >= limit:
                break

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)