import logging
import time
import cv2
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import pyrealsense2
try:
    import pyrealsense2 as rs
    REALSENSE_AVAILABLE = True
except ImportError:
    logger.warning("pyrealsense2 not installed. RealSense camera will not be available.")
    REALSENSE_AVAILABLE = False

class CameraStreamer:
    def __init__(self):
        self.pipeline = None
        self.profile = None
        self.cap = None
        self.mode = None # 'realsense', 'webcam', or 'mock'
        self.running = False
        self.overlays = []

    def start(self):
        logger.info("Starting CameraStreamer...")
        if self.running:
            logger.info("CameraStreamer already running.")
            return
        
        # Try RealSense first
        if REALSENSE_AVAILABLE:
            logger.info("Attempting to initialize RealSense camera...")
            try:
                # Check for devices before starting pipeline to avoid hanging
                ctx = rs.context()
                devices = ctx.query_devices()
                logger.info(f"Detected {len(devices)} RealSense device(s).")
                if len(devices) == 0:
                    raise RuntimeError("No RealSense devices detected.")

                logger.info("Creating RealSense pipeline...")
                self.pipeline = rs.pipeline()
                config = rs.config()
                logger.info("Configuring RealSense stream...")
                config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
                logger.info("Starting RealSense pipeline...")
                self.profile = self.pipeline.start(config)
                self.mode = 'realsense'
                logger.info("RealSense camera started successfully.")
                self.running = True
                return
            except Exception as e:
                logger.warning(f"Failed to start RealSense camera: {e}. Trying fallback.")
                self.pipeline = None
        else:
            logger.info("RealSense not available (import failed).")

        # Fallback to standard webcam
        logger.info("Attempting to initialize standard webcam...")
        try:
            logger.info("Opening VideoCapture(0)...")
            self.cap = cv2.VideoCapture(0)
            logger.info("Checking if webcam is opened...")
            if self.cap.isOpened():
                self.mode = 'webcam'
                logger.info("Webcam started successfully as fallback.")
                self.running = True
                return
            else:
                logger.warning("Webcam isOpened() returned False.")
                self.cap = None
        except Exception as e:
            logger.warning(f"Failed to start webcam: {e}. Using mock stream.")

        # Fallback to mock
        logger.info("Attempting to initialize mock stream...")
        self.mode = 'mock'
        logger.info("Using mock camera stream.")
        self.running = True

    def stop(self):
        if not self.running:
            return
        if self.mode == 'realsense' and self.pipeline:
            self.pipeline.stop()
            self.pipeline = None
            self.profile = None
        elif self.mode == 'webcam' and self.cap:
            self.cap.release()
            self.cap = None
        self.running = False
        self.mode = None
        logger.info("Camera stopped.")

    def set_overlays(self, overlays):
        self.overlays = overlays

    def clear_overlays(self):
        self.overlays = []

    def get_intrinsics(self):
        if not self.running:
            return None
        
        if self.mode == 'realsense' and self.profile:
            try:
                color_profile = self.profile.get_stream(rs.stream.color)
                intrin = color_profile.as_video_stream_profile().get_intrinsics()
                return {
                    "width": intrin.width,
                    "height": intrin.height,
                    "fx": intrin.fx,
                    "fy": intrin.fy,
                    "ppx": intrin.ppx,
                    "ppy": intrin.ppy,
                    "model": str(intrin.model),
                    "coeffs": intrin.coeffs
                }
            except Exception as e:
                logger.error(f"Failed to get RealSense intrinsics: {e}")
                return None
        elif self.mode in ('webcam', 'mock'):
            # Return some default intrinsics for fallback
            return {
                "width": 640,
                "height": 480,
                "fx": 600.0,
                "fy": 600.0,
                "ppx": 320.0,
                "ppy": 240.0,
                "model": "Plumb Bob",
                "coeffs": [0.0, 0.0, 0.0, 0.0, 0.0]
            }
        return None

    def _apply_overlays(self, frame):
        height, width = frame.shape[:2]
        frame = frame.copy()
        for u, v, label in self.overlays:
            u = max(0, min(1000, u))
            v = max(0, min(1000, v))
            x = int(u * width / 1000.0)
            y = int(v * height / 1000.0)
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(frame, label, (x + 10, y + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        return frame

    def get_frame(self):
        if not self.running:
            return None

        frame = None
        if self.mode == 'realsense':
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=1000)
                color_frame = frames.get_color_frame()
                if not color_frame:
                    return None
                frame = np.asanyarray(color_frame.get_data())
            except Exception as e:
                logger.error(f"Error reading from RealSense: {e}")
                return None

        elif self.mode == 'webcam':
            if self.cap:
                ret, r_frame = self.cap.read()
                if ret:
                    frame = r_frame
                else:
                    logger.error("Failed to read from webcam.")
                    return None

        elif self.mode == 'mock':
            img = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(img, f"Mock Stream: {time.time():.2f}", (50, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            noise = np.random.randint(0, 50, (480, 640, 3), dtype=np.uint8)
            img = cv2.add(img, noise)
            time.sleep(1/30.0)
            frame = img

        if frame is not None and self.overlays:
            frame = self._apply_overlays(frame)

        return frame
