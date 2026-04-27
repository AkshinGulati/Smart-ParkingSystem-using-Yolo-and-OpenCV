1. Project Title  
Smart Parking Detection System using YOLO and SORT Tracking  

2. Problem Statement  
Urban areas face a major issue of parking space management, where users often waste time searching for available parking slots. Manual monitoring is inefficient and error-prone.  

This project solves this by automatically detecting vehicles in a parking area using computer vision and determining real-time occupancy of parking slots from video feeds. It improves efficiency, reduces human effort, and enables smarter parking management.  

3. Role of Edge Computing  
This system is designed to run on edge devices such as NVIDIA Jetson Nano for real-time processing.  

Edge components:
- YOLO model for vehicle detection  
- SORT tracker for object tracking  
- Parking slot occupancy logic  
- Video frame processing pipeline  

Why edge computing:
- Reduces dependency on cloud servers  
- Enables real-time processing with low latency  
- Works offline without internet  
- Reduces bandwidth usage by processing video locally  

4. Methodology / Approach  
Pipeline:  
Input Video → Frame Preprocessing → YOLO Detection → SORT Tracking → Parking Slot Analysis → Output Display  

Stages:
- Input: Video stream of parking area  
- Preprocessing: Frame extraction and coordinate setup  
- Detection: YOLO detects vehicles in each frame  
- Tracking: SORT assigns unique IDs to vehicles  
- Output: Occupied and free parking slots displayed in real time  

5. Model Details  
- Model Type: YOLO (You Only Look Once) object detection model  
- Tracking: SORT (Simple Online and Realtime Tracking)  
- Input Format: RGB video frames  
- Input Size: 384 × 640 (model dependent)  
- Framework: Ultralytics YOLO   

Optimization:
- Lightweight YOLO variant used for real-time inference  
- Stream-based processing for efficiency  

6. Training Details  
- Dataset: Pretrained YOLO model trained on VisDrone dataset (vehicle detection dataset)
-  The VisDrone dataset is a large-scale benchmark dataset of drone-captured images and videos used for object detection and tracking.
-It includes annotated objects such as vehicles and pedestrians under diverse real-world conditions.
-It is commonly used to train and evaluate robust aerial vision models.
- Training Procedure:
  - Model trained using labeled vehicle images  
  - Bounding box annotations used for supervised learning  
  - Standard YOLO loss function used
    Image of FPS vs Frame graph
    https://drive.google.com/file/d/12HWCCeMJpSz82Wml_mpxIFqhDjYZoA5V/view?usp=drivesdk



7. Results / Output

Latency-75ms
Fps-13.2ms
YOLO-0.1ms
System Output:
- Real-time bounding boxes around detected vehicles  
- Unique tracking IDs for each vehicle  
- Parking slot occupancy detection (occupied/free)  
- FPS and latency metrics displayed on screen  

Performance:
- Real-time inference achieved on video input  
- FPS depends on hardware performance  
- Jetson Nano optimized for edge deployment  
- CPU version works but with higher latency
8. Setup Instructions
# Connect Jetson Nano to WiFi
nmcli device wifi list
nmcli device wifi connect "WIFI_NAME" password "PASSWORD"

# SSH connection to Jetson Nano
ssh username@jetson_ip

# Transfer project folder from PC to Jetson Nano
scp -r project_folder username@jetson_ip:/home/username/

# Port forwarding (localhost:8888)
ssh -L 8888:localhost:8888 nvidia@192.168.1.10

# Run the Python application
python main.py

# Basic Jetson / Linux commands
ls          # list files
cd folder   # change directory
pwd         # current directory path
mkdir name  # create new folder
rm file     # delete file
