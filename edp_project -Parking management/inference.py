# inference.py

import cv2
import time
import numpy as np
import cvzone

from ultralytics import YOLO
from sort import Sort

from config import *
from logger import log
from preprocessing import get_parking_polygons


def run_inference(video_path, polygon_option):

    log("Loading model...")
    model = YOLO(MODEL_PATH)

    log("Loading class names...")
    names = open(CLASSES_PATH).read().splitlines()

    cap = cv2.VideoCapture(video_path)

    tracker = Sort(max_age=TRACKER_MAX_AGE)

    # IMPORTANT: polygons are fixed here once
    raw_polys = get_parking_polygons(polygon_option)
    polys = [
        np.array(p, dtype=np.int32).reshape((-1, 1, 2))
        for p in raw_polys
    ]

    fps = 0

    while True:
        t0 = time.time()

        ret, frame = cap.read()
        if not ret:
            break

        # ---------------- YOLO ----------------
        t1 = time.time()
        results = model(frame, stream=True)
        yolo_ms = (time.time() - t1) * 1000

        # ---------------- detections ----------------
        dets = []

        for r in results:
            for b in r.boxes:
                cls_name = names[int(b.cls[0])]

                if cls_name == PREDICTION_CLASS:
                    x1, y1, x2, y2 = map(int, b.xyxy[0])
                    conf = float(b.conf[0])

                    dets.append([x1, y1, x2, y2, conf])

        dets = np.array(dets) if dets else np.empty((0, 5))

        # ---------------- tracking ----------------
        tracks = tracker.update(dets)
        occupied = set()

        for x1, y1, x2, y2, tid in tracks.astype(int):

            mid = (int((x1 + x2) // 2), int((y1 + y2) // 2))

            # ---------------- polygon check ----------------
            for i, p in enumerate(polys):

                if cv2.pointPolygonTest(p, mid, False) >= 0:
                    occupied.add(i)
                    break

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
            cvzone.putTextRect(frame, f"ID {tid}", (x1, y1 - 10))

        total = len(polys)

        # ---------------- stats ----------------
        cvzone.putTextRect(frame, f"Total: {total}", (20, 40))
        cvzone.putTextRect(frame, f"Occupied: {len(occupied)}", (20, 80))
        cvzone.putTextRect(frame, f"Free: {total - len(occupied)}", (20, 120))

        # ---------------- FPS ----------------
        dt = time.time() - t0
        fps = 0.9 * fps + 0.1 * (1 / dt)

        cv2.putText(frame, f"FPS: {fps:.2f}", (20, 160), 0, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Latency: {dt*1000:.1f} ms", (20, 200), 0, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"YOLO: {yolo_ms:.1f} ms", (20, 240), 0, 0.7, (0, 255, 255), 2)

        cv2.imshow("Parking Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()