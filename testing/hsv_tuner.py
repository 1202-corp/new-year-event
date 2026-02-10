"""HSV color range tuner with live sliders"""
import sys
import os
import platform

# Add parent directory to path to import game modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
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
    auto_exposure_values = [1.0, 0.25, 0.0] if system == "Windows" else [0.0, 0.25, 1.0]
    
    for auto_exp_val in auto_exposure_values:
        try:
            camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, auto_exp_val)
            return True
        except Exception:
            continue
    return False


def nothing(x):
    """Callback for trackbars"""
    pass


def main():
    """Main function"""
    # Initialize camera with platform-appropriate backend
    backend = get_camera_backend()
    camera = cv2.VideoCapture(Config.SNOWBALL_CAMERA_INDEX, backend)
    
    if not camera.isOpened():
        logger.error(f"Could not open camera {Config.SNOWBALL_CAMERA_INDEX}")
        return
    
    # Set camera format to MJPEG
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    camera.set(cv2.CAP_PROP_FOURCC, fourcc)
    
    # Set camera resolution
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
    
    logger.info("Camera initialized with settings from .env")
    
    # Create window for camera view
    window_name = "HSV Color Range Tuner"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)
    
    # Default HSV range (for white objects - wide range)
    # Hue: 0-179 (OpenCV uses 0-179, not 0-360)
    # Saturation: 0-255
    # Value: 0-255
    default_lower = np.array([0, 0, 200])    # Lower bound (darker white)
    default_upper = np.array([179, 30, 255])  # Upper bound (bright white)
    
    # Create trackbars for lower HSV bounds
    cv2.createTrackbar('H Min', window_name, default_lower[0], 179, nothing)
    cv2.createTrackbar('S Min', window_name, default_lower[1], 255, nothing)
    cv2.createTrackbar('V Min', window_name, default_lower[2], 255, nothing)
    
    # Create trackbars for upper HSV bounds
    cv2.createTrackbar('H Max', window_name, default_upper[0], 179, nothing)
    cv2.createTrackbar('S Max', window_name, default_upper[1], 255, nothing)
    cv2.createTrackbar('V Max', window_name, default_upper[2], 255, nothing)
    
    # Optional preprocessing
    cv2.createTrackbar('Use Blur', window_name, 1, 1, nothing)
    cv2.createTrackbar('Blur Size', window_name, 5, 30, nothing)
    
    # Morphological operations
    cv2.createTrackbar('Use Morph', window_name, 1, 1, nothing)
    cv2.createTrackbar('Morph Type', window_name, 2, 2, nothing)  # 0=OPEN, 1=CLOSE, 2=OPEN+CLOSE
    cv2.createTrackbar('Morph Size', window_name, 5, 20, nothing)
    
    logger.info("HSV Color Range Tuner initialized")
    logger.info("Adjust sliders to tune HSV color range")
    logger.info("Press 'q' to quit and print final values")
    logger.info("Press 'r' to reset to defaults")
    logger.info("Press 's' to save current settings")
    logger.info("Press 't' to toggle view mode")
    
    view_mode = 0  # 0 = side by side, 1 = mask only, 2 = masked original
    
    while True:
        ret, frame = camera.read()
        if not ret:
            logger.warning("Failed to read frame from camera")
            break
        
        # Read trackbar values
        h_min = cv2.getTrackbarPos('H Min', window_name)
        s_min = cv2.getTrackbarPos('S Min', window_name)
        v_min = cv2.getTrackbarPos('V Min', window_name)
        h_max = cv2.getTrackbarPos('H Max', window_name)
        s_max = cv2.getTrackbarPos('S Max', window_name)
        v_max = cv2.getTrackbarPos('V Max', window_name)
        
        use_blur = cv2.getTrackbarPos('Use Blur', window_name)
        blur_size = cv2.getTrackbarPos('Blur Size', window_name)
        
        use_morph = cv2.getTrackbarPos('Use Morph', window_name)
        morph_type = cv2.getTrackbarPos('Morph Type', window_name)
        morph_size = cv2.getTrackbarPos('Morph Size', window_name)
        
        # Ensure odd numbers for blur and morph sizes
        if blur_size % 2 == 0:
            blur_size += 1
        if blur_size < 1:
            blur_size = 1
        if morph_size % 2 == 0:
            morph_size += 1
        if morph_size < 1:
            morph_size = 1
        
        # Create HSV bounds
        lower_bound = np.array([h_min, s_min, v_min])
        upper_bound = np.array([h_max, s_max, v_max])
        
        # Convert BGR to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Optional blur
        if use_blur and blur_size > 1:
            hsv = cv2.GaussianBlur(hsv, (blur_size, blur_size), 0)
        
        # Create mask
        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        
        # Optional morphological operations
        if use_morph and morph_size > 1:
            kernel = np.ones((morph_size, morph_size), np.uint8)
            if morph_type == 0:  # OPEN
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            elif morph_type == 1:  # CLOSE
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            else:  # OPEN + CLOSE
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
                mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Apply mask to original frame
        masked_frame = cv2.bitwise_and(frame, frame, mask=mask)
        
        # Create output based on view mode
        if view_mode == 0:
            # Side by side: original, mask, masked
            mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            output_frame = np.hstack((frame, mask_colored, masked_frame))
        elif view_mode == 1:
            # Mask only
            output_frame = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        else:  # view_mode == 2
            # Masked original
            output_frame = masked_frame
        
        # Draw info overlay
        overlay = output_frame.copy()
        overlay_height = 280
        cv2.rectangle(overlay, (5, 5), (500, overlay_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, output_frame, 0.3, 0, output_frame)
        
        # Count white pixels in mask
        white_pixels = cv2.countNonZero(mask)
        total_pixels = mask.shape[0] * mask.shape[1]
        coverage_percent = (white_pixels / total_pixels) * 100
        
        morph_type_names = ["OPEN", "CLOSE", "OPEN+CLOSE"]
        view_mode_names = ["Side by Side", "Mask Only", "Masked Original"]
        
        info_text = [
            f"HSV Color Range:",
            f"Lower: H={h_min}, S={s_min}, V={v_min}",
            f"Upper: H={h_max}, S={s_max}, V={v_max}",
            "",
            f"Preprocessing:",
            f"Blur: {'ON' if use_blur else 'OFF'} ({blur_size}x{blur_size})",
            f"Morph: {'ON' if use_morph else 'OFF'} ({morph_type_names[morph_type]}, {morph_size}x{morph_size})",
            "",
            f"Mask Coverage: {coverage_percent:.2f}%",
            f"White Pixels: {white_pixels}",
            "",
            f"View Mode: {view_mode_names[view_mode]}",
            "",
            "Controls:",
            "Press 'q' to quit",
            "Press 'r' to reset",
            "Press 's' to save",
            "Press 't' to toggle view"
        ]
        
        y_offset = 25
        for i, text in enumerate(info_text):
            cv2.putText(output_frame, text, (10, y_offset + i * 18),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Display frame
        cv2.imshow(window_name, output_frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            # Reset to defaults
            cv2.setTrackbarPos('H Min', window_name, default_lower[0])
            cv2.setTrackbarPos('S Min', window_name, default_lower[1])
            cv2.setTrackbarPos('V Min', window_name, default_lower[2])
            cv2.setTrackbarPos('H Max', window_name, default_upper[0])
            cv2.setTrackbarPos('S Max', window_name, default_upper[1])
            cv2.setTrackbarPos('V Max', window_name, default_upper[2])
            cv2.setTrackbarPos('Use Blur', window_name, 1)
            cv2.setTrackbarPos('Blur Size', window_name, 5)
            cv2.setTrackbarPos('Use Morph', window_name, 1)
            cv2.setTrackbarPos('Morph Type', window_name, 2)
            cv2.setTrackbarPos('Morph Size', window_name, 5)
            logger.info("Settings reset to defaults")
        elif key == ord('s'):
            # Print current settings
            print("\n" + "="*60)
            print("CURRENT HSV COLOR RANGE SETTINGS:")
            print("="*60)
            print(f"Lower bound: H={h_min}, S={s_min}, V={v_min}")
            print(f"Upper bound: H={h_max}, S={s_max}, V={v_max}")
            print(f"\nPreprocessing:")
            print(f"use_blur: {bool(use_blur)}")
            print(f"blur_size: {blur_size}")
            print(f"\nMorphological operations:")
            print(f"use_morph: {bool(use_morph)}")
            print(f"morph_type: {morph_type_names[morph_type]} ({morph_type})")
            print(f"morph_size: {morph_size}")
            print(f"\nMask statistics:")
            print(f"Coverage: {coverage_percent:.2f}%")
            print(f"White pixels: {white_pixels}")
            print("\nPython code format:")
            print("-" * 60)
            print("import cv2")
            print("import numpy as np")
            print("")
            print("# Convert to HSV")
            print("hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)")
            print("")
            print("# Define color range")
            print(f"lower_bound = np.array([{h_min}, {s_min}, {v_min}])")
            print(f"upper_bound = np.array([{h_max}, {s_max}, {v_max}])")
            print("")
            print("# Create mask")
            print("mask = cv2.inRange(hsv, lower_bound, upper_bound)")
            if use_blur:
                print("")
                print("# Optional blur")
                print(f"hsv = cv2.GaussianBlur(hsv, ({blur_size}, {blur_size}), 0)")
                print("mask = cv2.inRange(hsv, lower_bound, upper_bound)")
            if use_morph:
                print("")
                print("# Morphological operations")
                print(f"kernel = np.ones(({morph_size}, {morph_size}), np.uint8)")
                if morph_type == 0:
                    print("mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)")
                elif morph_type == 1:
                    print("mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)")
                else:
                    print("mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)")
                    print("mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)")
            print("")
            print("# Apply mask")
            print("masked_frame = cv2.bitwise_and(frame, frame, mask=mask)")
            print("="*60 + "\n")
        elif key == ord('t'):
            view_mode = (view_mode + 1) % 3
            view_mode_names = ["Side by Side", "Mask Only", "Masked Original"]
            logger.info(f"View mode: {view_mode_names[view_mode]}")
    
    # Print final settings
    print("\n" + "="*60)
    print("FINAL HSV COLOR RANGE SETTINGS:")
    print("="*60)
    print(f"Lower bound: H={h_min}, S={s_min}, V={v_min}")
    print(f"Upper bound: H={h_max}, S={s_max}, V={v_max}")
    print(f"\nPreprocessing:")
    print(f"use_blur: {bool(use_blur)}")
    print(f"blur_size: {blur_size}")
    print(f"\nMorphological operations:")
    print(f"use_morph: {bool(use_morph)}")
    print(f"morph_type: {morph_type_names[morph_type]} ({morph_type})")
    print(f"morph_size: {morph_size}")
    print(f"\nMask statistics:")
    print(f"Coverage: {coverage_percent:.2f}%")
    print(f"White pixels: {white_pixels}")
    print("="*60 + "\n")
    
    # Cleanup
    camera.release()
    cv2.destroyAllWindows()
    logger.info("Camera released, windows closed")


if __name__ == "__main__":
    main()


