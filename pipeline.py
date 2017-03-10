import argparse

from moviepy.editor import VideoFileClip

from VehicleDetector import VehicleDetector

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Main entry for vehicle detection project")
    parser.add_argument("-test", action="store_true", help="Use test video")
    results = parser.parse_args()
    test = bool(results.test)

    if test:
        input_file = "test_video.mp4"
        output_file = "test.mp4"
    else:
        input_file = "project_video.mp4"
        output_file = "vehicle_detection.mp4"
    clip = VideoFileClip(input_file)
    vehicleDetector = VehicleDetector()
    output_clip = clip.fl_image(vehicleDetector.process_image)
    output_clip.write_videofile(output_file, audio=False)
