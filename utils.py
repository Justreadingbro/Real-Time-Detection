import socket
import time
import cv2


def get_local_ip():
    """Return a LAN IP (best-effort)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        # connect to a public DNS server — doesn't send data.
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = '127.0.0.1'
    finally:
        s.close()
    return ip


class FPSCounter:
    def __init__(self, alpha=0.9):
        self._last = None
        self._fps = 0.0
        self.alpha = alpha

    def tick(self):
        now = time.time()
        if self._last is None:
            self._last = now
            return 0.0
        dt = now - self._last
        self._last = now
        if dt == 0:
            return self._fps
        inst = 1.0 / dt
        # exponential moving average
        self._fps = (self.alpha * self._fps) + ((1 - self.alpha) * inst)
        return self._fps


class VideoWriterOptional:
    def __init__(self, path, fourcc_str='mp4v', fps=30):
        self.path = path
        self.fourcc = cv2.VideoWriter_fourcc(*fourcc_str)
        self.writer = None
        self.fps = fps

    def write(self, frame):
        h, w = frame.shape[:2]
        if self.writer is None:
            self.writer = cv2.VideoWriter(self.path, self.fourcc, self.fps, (w, h))
        self.writer.write(frame)

    def release(self):
        if self.writer:
            self.writer.release()