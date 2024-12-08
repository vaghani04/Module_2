import os
import numpy as np
from sentence_transformers import SentenceTransformer

# Suppress TensorFlow warnings and logging
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppresses INFO and WARNING logs

# Define the directory paths
TRANSCRIPTS_DIR = os.path.join('..', 'data', 'transcripts')
PROCESSED_DIR = os.path.join('..', 'data', 'processed')

# Load the SentenceTransformer model
print("Loading SentenceTransformer model...")
embed = SentenceTransformer('all-MiniLM-L6-v2')  # You can choose other pre-trained models
print("Model loaded successfully.")

def get_embeddings(sentences):
    """
    Generate embeddings for a list of sentences using SentenceTransformer.

    Args:
    - sentences (list): List of sentences for which embeddings are generated.

    Returns:
    - numpy.ndarray: Array of embeddings for each sentence.
    """
    embeddings = embed.encode(sentences, convert_to_numpy=True)
    return embeddings

def process_transcripts_and_generate_embeddings():
    """
    Process all transcript files in the transcripts directory, generate embeddings,
    and save them in the processed directory.
    """
    if not os.path.exists(PROCESSED_DIR):
        os.makedirs(PROCESSED_DIR)

    # Loop through all transcript files and generate embeddings
    for filename in os.listdir(TRANSCRIPTS_DIR):
        if filename.endswith('.txt'):
            file_path = os.path.join(TRANSCRIPTS_DIR, filename)

            # Read the content of the file
            with open(file_path, 'r', encoding='utf-8') as file:
                sentences = [line.strip() for line in file if line.strip()]

            # Generate embeddings for sentences
            embeddings = get_embeddings(sentences)

            # Save the embeddings to a corresponding processed file
            processed_file_path = os.path.join(PROCESSED_DIR, f"{filename.replace('.txt', '')}_embeddings.npy")
            with open(processed_file_path, 'wb') as emb_file:
                np.save(emb_file, embeddings)

            print(f"Embeddings for {filename} saved successfully.")

if __name__ == "__main__":
    process_transcripts_and_generate_embeddings()
