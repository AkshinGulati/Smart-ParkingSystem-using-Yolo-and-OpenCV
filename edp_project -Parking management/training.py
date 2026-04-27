"""
training.py

This module describes the training pipeline that can be used to train the YOLO-based object detection model used in this project.

The system is built using the Ultralytics YOLO framework, which supports training on custom datasets in a highly optimized and modular manner. In this project, a pretrained YOLO model trained on the VisDrone dataset has been utilized for detecting vehicles such as cars and vans in aerial or surveillance-style footage.
This specific system was trained on VisDrone dataset which is one of the best for drone captured images used for object detction and tracking.
If custom training were to be performed, the following pipeline would be used:
- Dataset preparation in YOLO format (images + labeled bounding boxes)
- Configuration of dataset YAML file (defining classes such as car, van, bicycle, pedestrian)
- Model initialization using a YOLO architecture (e.g., YOLOv8/YOLOv11 nano variant)
- Training using Ultralytics training API with hyperparameter tuning
- Validation on unseen parking surveillance frames
- Export of trained weights for inference deployment

This approach ensures high accuracy in real-time detection tasks while maintaining efficiency suitable for video-based parking management systems.
"""
