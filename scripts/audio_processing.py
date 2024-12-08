import os
import librosa
import soundfile as sf
from pyAudioAnalysis import audioSegmentation as aS
import warnings

warnings.filterwarnings("ignore", category=UserWarning)  # Ignore general UserWarnings
warnings.filterwarnings("ignore", category=FutureWarning)  # Ignore FutureWarnings


def segment_audio(input_audio, sr, start, end):
    """
    Extracts an audio segment from the input array between start and end times.
    """
    start_sample = int(start * sr)
    end_sample = int(end * sr)
    return input_audio[start_sample:end_sample]


def separate_speakers(audio_path, speaker1_output_path, speaker2_output_path):
    try:
        print(f"Loading audio file from: {audio_path}...")
        y, sr = librosa.load(audio_path, sr=16000)
        print("Audio loaded successfully.")
        
        print("Performing speaker diarization...")
        # Perform speaker diarization and get the result
        diarization_result, _, _ = aS.speaker_diarization(audio_path, 2)  # 2 speakers
        
        # Create output directories if they don't exist
        os.makedirs(os.path.dirname(speaker1_output_path), exist_ok=True)
        os.makedirs(os.path.dirname(speaker2_output_path), exist_ok=True)
        
        # Initialize audio containers for each speaker
        speaker1_audio = []
        speaker2_audio = []

        print("Splitting audio based on speaker diarization...")
        segment_duration = len(y) / sr
        segment_size = segment_duration / len(diarization_result)

        # Process diarization results to separate speakers
        for i, speaker in enumerate(diarization_result):
            start_time = i * segment_size
            end_time = start_time + segment_size
            
            if speaker == 0:
                speaker1_audio.extend(segment_audio(y, sr, start_time, end_time))
            elif speaker == 1:
                speaker2_audio.extend(segment_audio(y, sr, start_time, end_time))

        # Save separated audio
        if speaker1_audio:
            sf.write(speaker1_output_path, speaker1_audio, sr)
            print(f"Speaker 1 audio saved to: {speaker1_output_path}")
        else:
            print("No audio detected for Speaker 1.")
        
        if speaker2_audio:
            sf.write(speaker2_output_path, speaker2_audio, sr)
            print(f"Speaker 2 audio saved to: {speaker2_output_path}")
        else:
            print("No audio detected for Speaker 2.")

        print("Speaker separation and saving completed.")

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    # Define paths for input and output files
    base_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'audio')
    audio_path = os.path.join(base_path, 'main.wav')
    speaker1_output_path = os.path.join(base_path, 'speaker1.wav')
    speaker2_output_path = os.path.join(base_path, 'speaker2.wav')

    print(audio_path)
    print(speaker1_output_path)
    print(speaker2_output_path)

    if not os.path.exists(audio_path):
        print(f"Error: Audio file '{audio_path}' does not exist.")
    else:
        separate_speakers(audio_path, speaker1_output_path, speaker2_output_path)
