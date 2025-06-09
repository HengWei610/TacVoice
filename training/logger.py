# training/logger.py
import datetime

class Logger:
    def __init__(self):
        self.start_time = datetime.datetime.now()

    def log(self, message):
        time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"[{time}] {message}")
