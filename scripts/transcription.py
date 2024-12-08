import whisper
import os
import warnings


def transcribe_audio(audio_path, output_path):
    """
    Transcribes an audio file and saves the transcription to a text file.
    
    Parameters:
        audio_path (str): Path to the audio file.
        output_path (str): Path to save the transcription.
    """
    try:
        if not os.path.isfile(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        print(f"Loading Whisper model...")
        model = whisper.load_model("base")
        print(f"Transcribing audio file: {audio_path}...")
        result = model.transcribe(audio_path)
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Save transcription to file
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(result['text'])
        
        print(f"Transcription completed and saved to: {output_path}")
        
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

if __name__ == "__main__":
    
    # Suppress all warnings (optional)
    warnings.filterwarnings("ignore")

    # Suppress specific warnings, e.g., FP16 warnings
    warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")


    # Paths for Speaker 1 and Speaker 2 audio files
    main_audio_path = '..\data\\audio\\main.wav'
    speaker1_audio_path = '..\data\\audio\\speaker1.wav'
    speaker2_audio_path = '..\data\\audio\\speaker2.wav'
    
    # Paths for transcription output files
    main_transcription_path = '..\data\\transcripts\\main.txt'
    speaker1_transcription_path = '..\data\\transcripts\\speaker1.txt'
    speaker2_transcription_path = '..\data\\transcripts\\speaker2.txt'
    
    # Transcribe Main
    if os.path.exists(main_audio_path):
        transcribe_audio(main_audio_path, main_transcription_path)
    else:
        print(f"Error: Audio file '{main_audio_path}' does not exist.")

    # Transcribe Speaker 1
    if os.path.exists(speaker1_audio_path):
        transcribe_audio(speaker1_audio_path, speaker1_transcription_path)
    else:
        print(f"Error: Audio file '{speaker1_audio_path}' does not exist.")
    
    # Transcribe Speaker 2
    if os.path.exists(speaker2_audio_path):
        transcribe_audio(speaker2_audio_path, speaker2_transcription_path)
    else:
        print(f"Error: Audio file '{speaker2_audio_path}' does not exist.")
