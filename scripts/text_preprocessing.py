# src/preprocessing.py

import spacy
import os

# Load the spaCy model for sentence tokenization
nlp = spacy.load("en_core_web_sm")

def split_into_sentences(text):
    """
    This function splits the raw text into sentences using spaCy.

    Args:
    - text (str): Raw text input.

    Returns:
    - list: List of sentences.
    """
    # Process the text using spaCy
    doc = nlp(text)
    
    # Extract sentences from the processed text
    sentences = [sent.text.strip() for sent in doc.sents]
    
    return sentences

def read_text_from_file(file_path):
    """
    This function reads the content of a file.

    Args:
    - file_path (str): Path to the text file.

    Returns:
    - str: The content of the file.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

def save_sentences_to_file(sentences, output_file_path):
    """
    This function saves a list of sentences to a file.

    Args:
    - sentences (list): List of sentences to save.
    - output_file_path (str): Path to the output file.
    """
    os.makedirs(os.path.dirname(output_file_path), exist_ok=True)  # Ensure the directory exists
    with open(output_file_path, 'w', encoding='utf-8') as file:
        for sentence in sentences:
            file.write(sentence + "\n")

def process_and_save(input_file_path, output_file_path):
    """
    This function reads the text from the input file, splits it into sentences,
    and saves the sentences to an output file.

    Args:
    - input_file_path (str): Path to the input text file.
    - output_file_path (str): Path to the output file where sentences will be saved.
    """
    # Read text from the file
    text = read_text_from_file(input_file_path)
    
    # Split the text into sentences
    sentences = split_into_sentences(text)
    
    # Save the sentences to the output file
    save_sentences_to_file(sentences, output_file_path)

# Example usage
if __name__ == "__main__":
    # Define file paths
    main_input_file_path = "../data/transcripts/main.txt"
    input_file_path_1 = "../data/transcripts/speaker1.txt"
    input_file_path_2 = "../data/transcripts/speaker2.txt"

    main_output_file_path = "../data/processed/main_sentences.txt"
    output_file_path_1 = "../data/processed/speaker1_sentences.txt"
    output_file_path_2 = "../data/processed/speaker2_sentences.txt"
    
    # Process and save the sentences
    process_and_save(main_input_file_path, main_output_file_path)
    print(f"For main file: Sentences have been saved to {main_output_file_path}")

    process_and_save(input_file_path_1, output_file_path_1)
    print(f"For speaker-1: Sentences have been saved to {output_file_path_1}")

    process_and_save(input_file_path_2, output_file_path_2)
    print(f"For speaker-2: Sentences have been saved to {output_file_path_2}")
