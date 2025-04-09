import cv2
import numpy as np
import threading
from threading import Lock
import time

class ThreadedCamera:
    def __init__(self, camera_id=0, frame_width=640, frame_height=480):
        self.camera_id = camera_id
        self.frame_width = frame_width
        self.frame_height = frame_height
        
        # 初始化帧缓存
        self.frame_buffer = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
        self.lock = Lock()
        
        # 线程控制
        self.running = False
        self.thread = None
        
        # 初始化摄像头
        self.camera = cv2.VideoCapture(self.camera_id, cv2.CAP_DSHOW)
        # self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
        # self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        
        if not self.camera.isOpened():
            raise RuntimeError(f"无法打开摄像头: {camera_id}")
    
    def start(self):
        """启动摄像头读取线程"""
        if not self.running:
            self.running = True
            self.thread = threading.Thread(target=self._update_frame, daemon=True)
            self.thread.start()
    
    def stop(self):
        """停止摄像头读取线程"""
        self.running = False
        if self.thread is not None:
            self.thread.join()
            self.thread = None
    
    def _update_frame(self):
        """后台线程：持续读取摄像头帧"""
        while self.running:
            ret, frame = self.camera.read()
            if ret:
                frame = cv2.resize(frame, (self.frame_width, self.frame_height))
                with self.lock:
                    np.copyto(self.frame_buffer, frame)
            time.sleep(0.016)  # ~60fps
    
    def read(self):
        """读取当前帧（线程安全）"""
        with self.lock:
            return self.frame_buffer.copy()
    
    def release(self):
        """释放资源"""
        self.stop()
        if self.camera is not None:
            self.camera.release()
