# main.py

from inference import run_inference
from config import VIDEO_PATH

if __name__ == "__main__":
    polygon_option = "carpark_1"
    run_inference(VIDEO_PATH, polygon_option)