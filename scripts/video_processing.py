import ffmpeg
import os

def extract_audio(video_path, audio_output_path):
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(audio_output_path), exist_ok=True)
    
    # Extract audio using ffmpeg
    ffmpeg.input(video_path).output(audio_output_path, ac=1, ar=16000).run()
    print(f"Audio extracted successfully to: {audio_output_path}")

if __name__ == "__main__":
    # Input video path (relative path from scripts directory)
    video_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'interview_video.mp4')
    # Output audio path
    audio_output_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'audio', 'main.wav')
    
    # Check if the video file exists before proceeding
    if not os.path.exists(video_path):
        print(f"Error: Video file '{video_path}' does not exist.")
    else:
        extract_audio(video_path, audio_output_path)
