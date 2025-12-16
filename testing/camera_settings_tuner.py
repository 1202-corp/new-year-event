"""Camera settings tuner with live sliders"""
import sys
import os

# Add parent directory to path to import game modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from game.config import Config
from game.logger import get_logger

logger = get_logger()

# Global variables for trackbar callbacks
camera = None
current_settings = {}

def nothing(x):
    """Callback for trackbars (does nothing, settings are read directly)"""
    pass

def update_camera_setting(prop_id, value):
    """Update camera setting"""
    global camera, current_settings
    if camera is not None and camera.isOpened():
        try:
            camera.set(prop_id, value)
            current_settings[prop_id] = camera.get(prop_id)  # Get actual value (may differ from set)
        except Exception as e:
            logger.debug(f"Failed to set property {prop_id}: {e}")

def main():
    """Main function"""
    global camera, current_settings
    
    # Initialize camera
    camera = cv2.VideoCapture(Config.SNOWBALL_CAMERA_INDEX)
    
    if not camera.isOpened():
        logger.error(f"Could not open camera {Config.SNOWBALL_CAMERA_INDEX}")
        return
    
    # Set camera format to MJPEG
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    camera.set(cv2.CAP_PROP_FOURCC, fourcc)
    
    # Set camera resolution
    camera.set(cv2.CAP_PROP_FRAME_WIDTH, Config.SNOWBALL_CAMERA_WIDTH)
    camera.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.SNOWBALL_CAMERA_HEIGHT)
    
    # Get current values
    current_width = int(camera.get(cv2.CAP_PROP_FRAME_WIDTH))
    current_height = int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create window for camera view
    cv2.namedWindow('Camera Settings Tuner', cv2.WINDOW_NORMAL)
    
    # Create trackbars for various camera properties
    # Note: Some properties may not be supported by all cameras
    
    # Exposure (typical range: -13 to -1 for auto, 1-10000 for manual)
    exposure_min = -13
    exposure_max = 1
    exposure_default = int(camera.get(cv2.CAP_PROP_EXPOSURE))
    exposure_default = max(exposure_min, min(exposure_max, exposure_default))
    cv2.createTrackbar('Exposure', 'Camera Settings Tuner', 
                      exposure_default - exposure_min, 
                      exposure_max - exposure_min, nothing)
    
    # Brightness (typical range: 0-255 or 0-100)
    brightness_min = 0
    brightness_max = 255
    brightness_default = int(camera.get(cv2.CAP_PROP_BRIGHTNESS))
    brightness_default = max(brightness_min, min(brightness_max, brightness_default))
    cv2.createTrackbar('Brightness', 'Camera Settings Tuner', 
                       brightness_default, brightness_max, nothing)
    
    # Contrast (typical range: 0-255 or 0-100)
    contrast_min = 0
    contrast_max = 255
    contrast_default = int(camera.get(cv2.CAP_PROP_CONTRAST))
    contrast_default = max(contrast_min, min(contrast_max, contrast_default))
    cv2.createTrackbar('Contrast', 'Camera Settings Tuner', 
                       contrast_default, contrast_max, nothing)
    
    # Saturation (typical range: 0-255 or 0-100)
    saturation_min = 0
    saturation_max = 255
    saturation_default = int(camera.get(cv2.CAP_PROP_SATURATION))
    saturation_default = max(saturation_min, min(saturation_max, saturation_default))
    cv2.createTrackbar('Saturation', 'Camera Settings Tuner', 
                       saturation_default, saturation_max, nothing)
    
    # Hue (typical range: 0-255 or 0-100)
    hue_min = 0
    hue_max = 255
    hue_default = int(camera.get(cv2.CAP_PROP_HUE))
    hue_default = max(hue_min, min(hue_max, hue_default))
    cv2.createTrackbar('Hue', 'Camera Settings Tuner', 
                       hue_default, hue_max, nothing)
    
    # Gain (typical range: 0-100)
    gain_min = 0
    gain_max = 100
    gain_default = int(camera.get(cv2.CAP_PROP_GAIN))
    gain_default = max(gain_min, min(gain_max, gain_default))
    cv2.createTrackbar('Gain', 'Camera Settings Tuner', 
                       gain_default, gain_max, nothing)
    
    # Gamma (typical range: 0-500)
    gamma_min = 0
    gamma_max = 500
    gamma_default = int(camera.get(cv2.CAP_PROP_GAMMA))
    gamma_default = max(gamma_min, min(gamma_max, gamma_default))
    cv2.createTrackbar('Gamma', 'Camera Settings Tuner', 
                       gamma_default, gamma_max, nothing)
    
    # Sharpness (typical range: 0-255 or 0-100)
    sharpness_min = 0
    sharpness_max = 255
    sharpness_default = int(camera.get(cv2.CAP_PROP_SHARPNESS))
    sharpness_default = max(sharpness_min, min(sharpness_max, sharpness_default))
    cv2.createTrackbar('Sharpness', 'Camera Settings Tuner', 
                       sharpness_default, sharpness_max, nothing)
    
    # White Balance (typical range: 2800-6500 for temperature, or 0-100 for auto)
    wb_min = 0
    wb_max = 6500
    wb_default = int(camera.get(cv2.CAP_PROP_WHITE_BALANCE_BLUE_U))
    wb_default = max(wb_min, min(wb_max, wb_default))
    cv2.createTrackbar('White Balance', 'Camera Settings Tuner', 
                       wb_default, wb_max, nothing)
    
    # ISO Speed (if supported)
    iso_min = 0
    iso_max = 1600
    iso_default = 0
    try:
        iso_default = int(camera.get(cv2.CAP_PROP_ISO_SPEED))
        iso_default = max(iso_min, min(iso_max, iso_default))
    except:
        pass
    cv2.createTrackbar('ISO Speed', 'Camera Settings Tuner', 
                       iso_default, iso_max, nothing)
    
    logger.info("Camera Settings Tuner initialized")
    logger.info("Adjust sliders to tune camera settings")
    logger.info("Press 'q' to quit and print settings")
    logger.info("Press 'r' to reset to defaults")
    logger.info("Press 's' to save current settings")
    
    # Store initial settings for reset
    initial_settings = {
        'exposure': exposure_default,
        'brightness': brightness_default,
        'contrast': contrast_default,
        'saturation': saturation_default,
        'hue': hue_default,
        'gain': gain_default,
        'gamma': gamma_default,
        'sharpness': sharpness_default,
        'white_balance': wb_default,
        'iso': iso_default
    }
    
    while True:
        ret, frame = camera.read()
        if not ret:
            logger.warning("Failed to read frame from camera")
            break
        
        # Read trackbar values and update camera
        exposure_val = cv2.getTrackbarPos('Exposure', 'Camera Settings Tuner') + exposure_min
        update_camera_setting(cv2.CAP_PROP_EXPOSURE, exposure_val)
        
        brightness_val = cv2.getTrackbarPos('Brightness', 'Camera Settings Tuner')
        update_camera_setting(cv2.CAP_PROP_BRIGHTNESS, brightness_val)
        
        contrast_val = cv2.getTrackbarPos('Contrast', 'Camera Settings Tuner')
        update_camera_setting(cv2.CAP_PROP_CONTRAST, contrast_val)
        
        saturation_val = cv2.getTrackbarPos('Saturation', 'Camera Settings Tuner')
        update_camera_setting(cv2.CAP_PROP_SATURATION, saturation_val)
        
        hue_val = cv2.getTrackbarPos('Hue', 'Camera Settings Tuner')
        update_camera_setting(cv2.CAP_PROP_HUE, hue_val)
        
        gain_val = cv2.getTrackbarPos('Gain', 'Camera Settings Tuner')
        update_camera_setting(cv2.CAP_PROP_GAIN, gain_val)
        
        gamma_val = cv2.getTrackbarPos('Gamma', 'Camera Settings Tuner')
        update_camera_setting(cv2.CAP_PROP_GAMMA, gamma_val)
        
        sharpness_val = cv2.getTrackbarPos('Sharpness', 'Camera Settings Tuner')
        update_camera_setting(cv2.CAP_PROP_SHARPNESS, sharpness_val)
        
        wb_val = cv2.getTrackbarPos('White Balance', 'Camera Settings Tuner')
        update_camera_setting(cv2.CAP_PROP_WHITE_BALANCE_BLUE_U, wb_val)
        
        iso_val = cv2.getTrackbarPos('ISO Speed', 'Camera Settings Tuner')
        try:
            update_camera_setting(cv2.CAP_PROP_ISO_SPEED, iso_val)
        except:
            pass
        
        # Draw info on frame
        info_text = [
            f"Resolution: {current_width}x{current_height}",
            f"Exposure: {exposure_val}",
            f"Brightness: {brightness_val}",
            f"Contrast: {contrast_val}",
            f"Saturation: {saturation_val}",
            f"Gain: {gain_val}",
            "",
            "Press 'q' to quit and print settings",
            "Press 'r' to reset",
            "Press 's' to save"
        ]
        
        y_offset = 30
        for i, text in enumerate(info_text):
            cv2.putText(frame, text, (10, y_offset + i * 25),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Display frame
        cv2.imshow('Camera Settings Tuner', frame)
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('r'):
            # Reset to initial settings
            cv2.setTrackbarPos('Exposure', 'Camera Settings Tuner', 
                              initial_settings['exposure'] - exposure_min)
            cv2.setTrackbarPos('Brightness', 'Camera Settings Tuner', 
                              initial_settings['brightness'])
            cv2.setTrackbarPos('Contrast', 'Camera Settings Tuner', 
                              initial_settings['contrast'])
            cv2.setTrackbarPos('Saturation', 'Camera Settings Tuner', 
                              initial_settings['saturation'])
            cv2.setTrackbarPos('Hue', 'Camera Settings Tuner', 
                              initial_settings['hue'])
            cv2.setTrackbarPos('Gain', 'Camera Settings Tuner', 
                              initial_settings['gain'])
            cv2.setTrackbarPos('Gamma', 'Camera Settings Tuner', 
                              initial_settings['gamma'])
            cv2.setTrackbarPos('Sharpness', 'Camera Settings Tuner', 
                              initial_settings['sharpness'])
            cv2.setTrackbarPos('White Balance', 'Camera Settings Tuner', 
                              initial_settings['white_balance'])
            cv2.setTrackbarPos('ISO Speed', 'Camera Settings Tuner', 
                              initial_settings['iso'])
            logger.info("Settings reset to initial values")
        elif key == ord('s'):
            # Save current settings (print them)
            print_settings(camera)
    
    # Print final settings
    print("\n" + "="*60)
    print("FINAL CAMERA SETTINGS:")
    print("="*60)
    print_settings(camera)
    print("="*60 + "\n")
    
    # Cleanup
    camera.release()
    cv2.destroyAllWindows()
    logger.info("Camera released, windows closed")

def print_settings(camera):
    """Print all camera settings"""
    if camera is None or not camera.isOpened():
        return
    
    settings = {
        'Frame Width': camera.get(cv2.CAP_PROP_FRAME_WIDTH),
        'Frame Height': camera.get(cv2.CAP_PROP_FRAME_HEIGHT),
        'FPS': camera.get(cv2.CAP_PROP_FPS),
        'FourCC': int(camera.get(cv2.CAP_PROP_FOURCC)),
        'Exposure': camera.get(cv2.CAP_PROP_EXPOSURE),
        'Brightness': camera.get(cv2.CAP_PROP_BRIGHTNESS),
        'Contrast': camera.get(cv2.CAP_PROP_CONTRAST),
        'Saturation': camera.get(cv2.CAP_PROP_SATURATION),
        'Hue': camera.get(cv2.CAP_PROP_HUE),
        'Gain': camera.get(cv2.CAP_PROP_GAIN),
        'Gamma': camera.get(cv2.CAP_PROP_GAMMA),
        'Sharpness': camera.get(cv2.CAP_PROP_SHARPNESS),
        'White Balance Blue U': camera.get(cv2.CAP_PROP_WHITE_BALANCE_BLUE_U),
        'White Balance Red V': camera.get(cv2.CAP_PROP_WHITE_BALANCE_RED_V),
    }
    
    # Try to get ISO if supported
    try:
        settings['ISO Speed'] = camera.get(cv2.CAP_PROP_ISO_SPEED)
    except:
        settings['ISO Speed'] = 'Not supported'
    
    # Convert FourCC to string
    fourcc_int = int(settings['FourCC'])
    fourcc_str = "".join([chr((fourcc_int >> 8 * i) & 0xFF) for i in range(4)])
    settings['FourCC'] = f"{fourcc_str} ({fourcc_int})"
    
    # Print settings
    for key, value in settings.items():
        print(f"{key:25s}: {value}")
    
    # Print Python code format
    print("\nPython code format:")
    print("-" * 60)
    print("# Camera settings")
    print(f"camera.set(cv2.CAP_PROP_EXPOSURE, {settings['Exposure']})")
    print(f"camera.set(cv2.CAP_PROP_BRIGHTNESS, {settings['Brightness']})")
    print(f"camera.set(cv2.CAP_PROP_CONTRAST, {settings['Contrast']})")
    print(f"camera.set(cv2.CAP_PROP_SATURATION, {settings['Saturation']})")
    if settings['Hue'] != 0:
        print(f"camera.set(cv2.CAP_PROP_HUE, {settings['Hue']})")
    if settings['Gain'] != 0:
        print(f"camera.set(cv2.CAP_PROP_GAIN, {settings['Gain']})")
    if settings['Gamma'] != 0:
        print(f"camera.set(cv2.CAP_PROP_GAMMA, {settings['Gamma']})")
    if settings['Sharpness'] != 0:
        print(f"camera.set(cv2.CAP_PROP_SHARPNESS, {settings['Sharpness']})")
    if isinstance(settings['ISO Speed'], (int, float)) and settings['ISO Speed'] != 0:
        print(f"camera.set(cv2.CAP_PROP_ISO_SPEED, {settings['ISO Speed']})")
    print("-" * 60)


if __name__ == "__main__":
    main()

