from components.detectors import MultiCameraPeopleDetector
from components.cameras import Camera

# Camera addresses: These are the addresses of the cameras that will be used for the demo. 
# Please try it with all formats of streams like HLS etc. Ideally the pipeline should
# work with all formats of cameras. 

camera_addresses = ["rtsp://3.76.45.126:8554/stream1", "rtsp://18.185.101.169:8554/stream1"]

cameras = []

for i, camera_address in enumerate(camera_addresses):
    cameras.append(Camera(i, camera_address))

# Create a detector object and pass the cameras to it.
detector = MultiCameraPeopleDetector(cameras)

detector.play()