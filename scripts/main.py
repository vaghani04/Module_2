import os
import subprocess
import sys

def run_script(script_name):
    try:
        base_path = os.path.dirname(__file__)
        # scripts_path = os.path.join(base_path, 'scripts')
        script_path = os.path.join(base_path, f'{script_name}.py')

        print(f"Running {script_name}.py...")
        result = subprocess.run([sys.executable, script_path], check=True)
        if result.returncode == 0:
            print(f"{script_name}.py executed successfully.")
        else:
            print(f"Error: {script_name}.py execution failed.")
            sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"An error occurred while executing {script_name}.py: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while executing {script_name}.py: {e}")
        sys.exit(1)

def main():
    script_files = [
        'video_processing',
        'audio_processing',
        'transcription',
        'text_preprocessing',
        'embeddings_and_pinecone_store',
        'communication_style_summary',
        'active_listening_summary',
        'engagement_summary',
        'rag_summary_generating'
    ]
    
    for script in script_files:
        run_script(script)

if __name__ == "__main__":
    main()
