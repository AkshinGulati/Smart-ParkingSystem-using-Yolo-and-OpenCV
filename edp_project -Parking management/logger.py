# logger.py

import time

def log(msg):
    print(f"[LOG {time.strftime('%H:%M:%S')}] {msg}")