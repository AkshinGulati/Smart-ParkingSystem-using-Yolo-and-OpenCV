import cv2, time, numpy as np
from ultralytics import YOLO
import cvzone
from sort import Sort
from car_parking_coordinate_data import car_park_coordinate

def run_detection(video_path, polygon_option):
    cap = cv2.VideoCapture(video_path)

    tracker = Sort(max_age=30)
    model = YOLO('yolov11n_visdrone_model.pt')
    names = open('classes.txt').read().splitlines()
    polys = car_park_coordinate(polygon_option)

    fps = 0

    while True:
        t0 = time.time()
        ret, frame = cap.read()
        if not ret: break

        # -------- YOLO --------
        t1 = time.time()
        results = model(frame, stream=True)
        yolo_ms = (time.time() - t1) * 1000

        # -------- detections --------
        dets = []
        for r in results:
            for b in r.boxes:
                if names[int(b.cls[0])] == "car":
                    x1,y1,x2,y2 = map(int, b.xyxy[0])
                    dets.append([x1,y1,x2,y2,float(b.conf[0])])
        dets = np.array(dets) if dets else np.empty((0,5))

        tracks = tracker.update(dets)
        occ = set()

        # -------- tracking + polygon check --------
        for x1,y1,x2,y2,tid in tracks.astype(int):
            mid = (int((x1+x2)//2), int((y1+y2)//2))  # ✅ FIXED HERE

            for i, p in enumerate(polys):
                if cv2.pointPolygonTest(p, mid, False) >= 0:
                    occ.add(i)
                    break

            cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)
            cvzone.putTextRect(frame,f'ID {tid}',(x1,y1-10))

        total = len(polys)

        # -------- stats --------
        cvzone.putTextRect(frame,f"Total: {total}",(20,40))
        cvzone.putTextRect(frame,f"Occupied: {len(occ)}",(20,80))
        cvzone.putTextRect(frame,f"Free: {total-len(occ)}",(20,120))

        # -------- FPS + latency --------
        dt = time.time() - t0
        fps = 0.9*fps + 0.1*(1/dt)

        cv2.putText(frame,f"FPS: {fps:.2f}",(20,160),0,0.7,(0,255,0),2)
        cv2.putText(frame,f"Latency: {dt*1000:.1f} ms",(20,200),0,0.7,(0,255,0),2)
        cv2.putText(frame,f"YOLO: {yolo_ms:.1f} ms",(20,240),0,0.7,(0,255,255),2)

        cv2.imshow("Parking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()