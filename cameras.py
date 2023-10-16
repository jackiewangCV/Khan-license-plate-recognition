from collections import deque
import time


class Camera:
    def __init__(self, id, cam_address):
        self.internal_id = time.time_ns()
        self.id = id
        self.cam_address = cam_address

        self.imageq = deque(maxlen=100)
        self.countq = deque(maxlen=60)
        self.positionq = deque(maxlen=60)