"""Timed camera capture - takes photos every 0.3 seconds"""
import sys
import os
import platform
import time
from datetime import datetime

# Add parent directory to path to import game modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
from game.config import Config
from game.logger import get_logger

logger = get_logger()


def get_camera_backend():
    """Get appropriate camera backend based on platform"""
    system = platform.system()
    if system == "Windows":
        return cv2.CAP_DSHOW
    elif system == "Linux":
        return cv2.CAP_V4L2
    else:
        return 0


def setup_camera_exposure(camera):
    """Setup camera exposure with correct AUTO_EXPOSURE value based on backend"""
    system = platform.system()
    if system == "Windows":
        # Для DirectShow: 1.0 или 0.25 = ручной режим
        auto_exposure_values = [1.0, 0.25, 0.0]
    else:
        # Для Linux/V4L2: 0.25 = ручной режим (не 0.0!)
        auto_exposure_values = [0.25, 0.0, 1.0]
    
    for auto_exp_val in auto_exposure_values:
        try:
            result = camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, auto_exp_val)
            if result:
                logger.info(f"Установлен AUTO_EXPOSURE = {auto_exp_val}")
                return True
        except Exception:
            continue
    
    logger.warning("Не удалось установить AUTO_EXPOSURE")
    return False


def main():
    """Main function"""
    # Capture interval in seconds
    CAPTURE_INTERVAL = 0.05
    
    # Create output directory
    output_dir = "captures"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize camera with platform-appropriate backend
    backend = get_camera_backend()
    camera = cv2.VideoCapture(Config.SNOWBALL_CAMERA_INDEX, backend)
    
    if not camera.isOpened():
        logger.error(f"Could not open camera {Config.SNOWBALL_CAMERA_INDEX}")
        return
    
    logger.info(f"Opening camera {Config.SNOWBALL_CAMERA_INDEX} with backend {backend}")
    
    # Set camera format to MJPEG (must be done before setting resolution)
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    camera.set(cv2.CAP_PROP_FOURCC, fourcc)
    
    # Verify MJPEG format was set
    current_fourcc = int(camera.get(cv2.CAP_PROP_FOURCC))
    fourcc_str = "".join([chr((current_fourcc >> 8 * i) & 0xFF) for i in range(4)])
    logger.info(f"Camera format: {fourcc_str} (requested: MJPG)")
    
    # Set camera resolution from .env
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, Config.SNOWBALL_CAMERA_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.SNOWBALL_CAMERA_HEIGHT)
    
    # Setup exposure control (must be done before setting exposure value)
    setup_camera_exposure(camera)
    
    # Apply camera quality settings from config
    camera.set(cv2.CAP_PROP_EXPOSURE, Config.SNOWBALL_CAMERA_EXPOSURE)
    camera.set(cv2.CAP_PROP_BRIGHTNESS, Config.SNOWBALL_CAMERA_BRIGHTNESS)
    camera.set(cv2.CAP_PROP_CONTRAST, Config.SNOWBALL_CAMERA_CONTRAST)
    camera.set(cv2.CAP_PROP_SATURATION, Config.SNOWBALL_CAMERA_SATURATION)
    camera.set(cv2.CAP_PROP_SHARPNESS, Config.SNOWBALL_CAMERA_SHARPNESS)
    camera.set(cv2.CAP_PROP_GAIN, Config.SNOWBALL_CAMERA_GAIN)
    camera.set(cv2.CAP_PROP_FOCUS, Config.SNOWBALL_CAMERA_FOCUS)
    
    # Disable autofocus when setting manual focus
    if Config.SNOWBALL_CAMERA_FOCUS >= 0:
        try:
            camera.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        except Exception:
            pass
    
    # Read a few frames to clear buffer and apply settings
    logger.info("Clearing camera buffer...")
    for i in range(5):
        ret, _ = camera.read()
        if not ret:
            logger.warning(f"Failed to read frame {i+1} during buffer clearing")
            break
    
    # Verify actual camera settings
    actual_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_exposure = camera.get(cv2.CAP_PROP_EXPOSURE)
    actual_brightness = camera.get(cv2.CAP_PROP_BRIGHTNESS)
    
    logger.info("=" * 60)
    logger.info("Camera settings from .env:")
    logger.info(f"  Resolution: {Config.SNOWBALL_CAMERA_WIDTH}x{Config.SNOWBALL_CAMERA_HEIGHT}")
    logger.info(f"  Format: MJPEG (MJPG)")
    logger.info(f"  Exposure: {Config.SNOWBALL_CAMERA_EXPOSURE}")
    logger.info(f"  Brightness: {Config.SNOWBALL_CAMERA_BRIGHTNESS}")
    logger.info(f"  Contrast: {Config.SNOWBALL_CAMERA_CONTRAST}")
    logger.info(f"  Saturation: {Config.SNOWBALL_CAMERA_SATURATION}")
    logger.info(f"  Sharpness: {Config.SNOWBALL_CAMERA_SHARPNESS}")
    logger.info(f"  Gain: {Config.SNOWBALL_CAMERA_GAIN}")
    logger.info(f"  Focus: {Config.SNOWBALL_CAMERA_FOCUS}")
    logger.info("=" * 60)
    logger.info("Actual camera settings:")
    logger.info(f"  Resolution: {actual_width}x{actual_height}")
    logger.info(f"  Format: {fourcc_str}")
    logger.info(f"  Exposure: {actual_exposure}")
    logger.info(f"  Brightness: {actual_brightness}")
    logger.info("=" * 60)
    
    # Verify resolution matches
    if actual_width != Config.SNOWBALL_CAMERA_WIDTH or actual_height != Config.SNOWBALL_CAMERA_HEIGHT:
        logger.warning(f"Resolution mismatch! Requested: {Config.SNOWBALL_CAMERA_WIDTH}x{Config.SNOWBALL_CAMERA_HEIGHT}, "
                      f"Actual: {actual_width}x{actual_height}")
    
    # Verify format matches
    if fourcc_str != "MJPG":
        logger.warning(f"Format mismatch! Requested: MJPG, Actual: {fourcc_str}")
    
    logger.info(f"Capture interval: {CAPTURE_INTERVAL} seconds")
    logger.info(f"Output directory: {output_dir}/")
    logger.info("Press 'q' to quit, SPACE to pause/resume")
    
    # Create window for preview
    window_name = "Timed Capture - Press 'q' to quit, SPACE to pause"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)
    
    frame_count = 0
    last_capture_time = time.time()
    is_paused = False
    
    while True:
        ret, frame = camera.read()
        if not ret:
            logger.warning("Failed to read frame from camera")
            break
        
        current_time = time.time()
        time_since_last_capture = current_time - last_capture_time
        
        # Check if it's time to capture (only if not paused)
        if not is_paused and time_since_last_capture >= CAPTURE_INTERVAL:
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]  # milliseconds precision
            filename = os.path.join(output_dir, f"capture_{timestamp}.jpg")
            
            # Save frame
            cv2.imwrite(filename, frame)
            frame_count += 1
            
            logger.info(f"Captured frame #{frame_count}: {filename}")
            last_capture_time = current_time
        
        # Create display frame with overlay
        display_frame = frame.copy()
        
        # Draw overlay with info
        overlay_height = 120
        cv2.rectangle(display_frame, (5, 5), (500, overlay_height), (0, 0, 0), -1)
        
        # Calculate time until next capture
        time_until_next = CAPTURE_INTERVAL - time_since_last_capture
        if time_until_next < 0:
            time_until_next = 0
        
        status_text = "PAUSED" if is_paused else "RUNNING"
        info_text = [
            f"Timed Capture - Every {CAPTURE_INTERVAL}s",
            f"Status: {status_text}",
            f"Frames captured: {frame_count}",
            f"Time until next: {time_until_next:.2f}s" if not is_paused else "Paused",
            "",
            "Press 'q' to quit, SPACE to pause/resume"
        ]
        
        y_offset = 25
        for i, text in enumerate(info_text):
            cv2.putText(display_frame, text, (10, y_offset + i * 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Draw progress bar for next capture
        bar_width = 400
        bar_height = 20
        bar_x = 10
        bar_y = overlay_height + 10
        progress = min(time_since_last_capture / CAPTURE_INTERVAL, 1.0)
        
        # Background bar
        cv2.rectangle(display_frame, (bar_x, bar_y), 
                     (bar_x + bar_width, bar_y + bar_height), (50, 50, 50), -1)
        # Progress bar
        cv2.rectangle(display_frame, (bar_x, bar_y), 
                     (int(bar_x + bar_width * progress), bar_y + bar_height), 
                     (0, 255, 0), -1)
        # Border
        cv2.rectangle(display_frame, (bar_x, bar_y), 
                     (bar_x + bar_width, bar_y + bar_height), (255, 255, 255), 2)
        
        # Display frame
        cv2.imshow(window_name, display_frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):  # Spacebar to pause/resume
            is_paused = not is_paused
            if is_paused:
                logger.info("Capture PAUSED")
            else:
                logger.info("Capture RESUMED")
                # Reset timer when resuming
                last_capture_time = time.time()
    
    # Cleanup
    camera.release()
    cv2.destroyAllWindows()
    logger.info(f"Capture stopped. Total frames captured: {frame_count}")
    logger.info(f"All images saved to: {output_dir}/")


if __name__ == "__main__":
    main()

