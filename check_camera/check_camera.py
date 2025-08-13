import pyrealsense2 as rs
import time
import cv2
import numpy as np

def check_realsense_device():
    """
    Checks for connected RealSense devices and prints their information.
    This is the most basic test to ensure the library and drivers can see the camera.
    """
    print("--- Step 1: Checking for RealSense devices ---")
    try:
        ctx = rs.context()
        devices = ctx.query_devices()
        
        if len(devices) == 0:
            print("❌ RESULT: No RealSense devices found.")
            print("   TROUBLESHOOTING:")
            print("   - Ensure the camera is securely connected to a USB 3 port.")
            print("   - Try a different USB port or cable.")
            print("   - Check if the camera is being used by another program.")
            return False # Indicate failure

        print(f"✅ RESULT: Found {len(devices)} RealSense device(s).")
        for i, dev in enumerate(devices):
            name = dev.get_info(rs.camera_info.name)
            serial = dev.get_info(rs.camera_info.serial_number)
            fw_version = dev.get_info(rs.camera_info.firmware_version)
            print(f"\n--- Device #{i+1} ---")
            print(f"   Name:                 {name}")
            print(f"   Serial Number:        {serial}")
            print(f"   Firmware Version:     {fw_version}")
        return True # Indicate success
    except Exception as e:
        print(f"💥 An unexpected error occurred in Step 1: {e}")
        return False # Indicate failure

def check_pipeline_start():
    """
    Step 2: Tries to initialize the pipeline and fetch a single frame.
    This tests the ability to actually start the stream and get data.
    """
    print("\n--- Step 2: Testing camera pipeline start and frame acquisition ---")
    pipeline = None
    try:
        pipeline = rs.pipeline()
        config = rs.config()
        
        # Request a specific stream configuration
        width, height, fps = 1280, 720, 30
        print(f"   Configuring stream: Color {width}x{height} @ {fps}fps...")
        config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        
        print("   Attempting to start the pipeline...")
        profile = pipeline.start(config)
        print("   ✅ Pipeline started successfully.")
        
        device = profile.get_device()
        print(f"   Pipeline is using device: {device.get_info(rs.camera_info.name)}")
        
        # Warm-up phase to allow auto-exposure to settle
        print("   Warming up camera for auto-exposure...")
        for _ in range(30):
            pipeline.wait_for_frames() # Discard the first 30 frames
        print("   ✅ Camera is ready.")

        print("   Attempting to wait for frames (timeout: 5 seconds)...")
        frames = pipeline.wait_for_frames(5000) # 5-second timeout
        
        color_frame = frames.get_color_frame()
        if not color_frame:
            print("   ❌ RESULT: Pipeline started, but no color frame was received.")
        else:
            print(f"   ✅ RESULT: Successfully received a color frame!")
            print(f"      - Resolution: {color_frame.get_width()}x{color_frame.get_height()}")
            print(f"      - Timestamp: {color_frame.get_timestamp()} ms")
            
            # Convert to numpy array and save
            try:
                image = np.asanyarray(color_frame.get_data())
                filename = "debug_image.jpg"
                cv2.imwrite(filename, image)
                print(f"   ✅ Image saved as '{filename}'")
            except Exception as e:
                print(f"   ❌ Failed to save image: {e}")

    except Exception as e:
        print(f"   💥 An error occurred during Step 2: {e}")
        print("      This is likely the point of failure in the main application.")
    finally:
        if pipeline:
            print("   Stopping the pipeline...")
            pipeline.stop()
            print("   Pipeline stopped.")


if __name__ == "__main__":
    if check_realsense_device():
        # Only proceed to step 2 if step 1 was successful
        check_pipeline_start() 