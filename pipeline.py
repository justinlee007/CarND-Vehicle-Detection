from moviepy.editor import VideoFileClip

from VehicleDetector import VehicleDetector

if __name__ == "__main__":
    vehicleDetector = VehicleDetector()
    output_file = "test.mp4"
    # output_file = "vehicle_detection.mp4"
    clip = VideoFileClip("test_video.mp4")
    # clip = VideoFileClip("project_video.mp4")
    output_clip = clip.fl_image(vehicleDetector.process_image)
    output_clip.write_videofile(output_file, audio=False)
