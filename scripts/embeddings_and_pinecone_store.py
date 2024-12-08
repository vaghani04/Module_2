import os
import pinecone
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import warnings
from pinecone import Pinecone, ServerlessSpec

warnings.filterwarnings("ignore", category=UserWarning)

# Load environment variables from .env
load_dotenv()

def init_pinecone(api_key, environment):
    """
    Initialize the Pinecone environment and create indexes for interviewer, candidate, and main if they don't exist.

    Args:
    - api_key (str): Pinecone API key.
    - environment (str): Pinecone environment name.
    """
    # Initialize Pinecone
    pc = Pinecone(api_key=api_key)

    # Define index names
    interviewer_index_name = "interviewer-index"
    candidate_index_name = "candidate-index"
    main_index_name = "main-index"
    
    # Define index spec (dimension and metric)
    index_spec = ServerlessSpec(cloud="aws", region="us-east-1")
    
    # Check and create the indexes if they don't exist
    if interviewer_index_name not in pc.list_indexes().names():
        print(f"Creating index: {interviewer_index_name}")
        pc.create_index(interviewer_index_name, dimension=384, spec=index_spec)  # Add spec here
    
    if candidate_index_name not in pc.list_indexes().names():
        print(f"Creating index: {candidate_index_name}")
        pc.create_index(candidate_index_name, dimension=384, spec=index_spec)
    
    if main_index_name not in pc.list_indexes().names():
        print(f"Creating index: {main_index_name}")
        pc.create_index(main_index_name, dimension=384, spec=index_spec)
    
    # Return the index objects for all three
    interviewer_index = pc.Index(interviewer_index_name)
    candidate_index = pc.Index(candidate_index_name)
    main_index = pc.Index(main_index_name)
    
    return interviewer_index, candidate_index, main_index

def get_embeddings(sentences):
    """
    Generate embeddings for a list of sentences using SentenceTransformer.

    Args:
    - sentences (list): List of sentences for which embeddings are generated.

    Returns:
    - numpy.ndarray: Array of embeddings for each sentence.
    """
    embed = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = embed.encode(sentences, convert_to_numpy=True)
    return embeddings

def store_embeddings_in_pinecone(index, sentences, embeddings):
    """
    Store the sentence embeddings in Pinecone.

    Args:
    - index (pinecone.Index): The Pinecone index.
    - sentences (list): The list of sentences.
    - embeddings (list): The list of sentence embeddings.
    """
    # Prepare data for upsert operation
    upsert_data = [(str(i), embedding.tolist(), {"sentence": sentence}) 
                   for i, (embedding, sentence) in enumerate(zip(embeddings, sentences))]
    
    # Perform the upsert operation (store embeddings)
    print(f"Upserting {len(upsert_data)} vectors to Pinecone...")
    index.upsert(vectors=upsert_data)
    print("Upsert complete!")

def process_transcripts_and_store_embeddings():
    """
    Process the speaker1.txt and speaker2.txt to decide candidate and interviewer, generate embeddings,
    and store them in Pinecone.
    """
    # Initialize Pinecone
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")  # Load API key from .env
    environment = "us-east-1"
    interviewer_index, candidate_index, main_index = init_pinecone(PINECONE_API_KEY, environment)

    # Define the transcripts directory
    PROCESSED_DIR = os.path.join('..', 'data', 'processed')

    # Read the contents of speaker1.txt and speaker2.txt
    with open(os.path.join(PROCESSED_DIR, 'speaker1_sentences.txt'), 'r', encoding='utf-8') as file:
        speaker1_data = file.read()
    
    with open(os.path.join(PROCESSED_DIR, 'speaker2_sentences.txt'), 'r', encoding='utf-8') as file:
        speaker2_data = file.read()

    # Assign the candidate and interviewer based on the length of the data
    if len(speaker1_data) > len(speaker2_data):
        candidate_data = speaker1_data
        interviewer_data = speaker2_data
    else:
        candidate_data = speaker2_data
        interviewer_data = speaker1_data

    # Read the content of the main.txt file for the whole conversation
    with open(os.path.join(PROCESSED_DIR, 'main_sentences.txt'), 'r', encoding='utf-8') as file:
        main_data = file.read()

    # Generate embeddings for each group of sentences
    # Split into sentences for better processing
    interviewer_sentences = interviewer_data.split('\n')
    candidate_sentences = candidate_data.split('\n')
    main_sentences = main_data.split('\n')

    # Generate embeddings for each group of sentences
    interviewer_embeddings = get_embeddings(interviewer_sentences)
    candidate_embeddings = get_embeddings(candidate_sentences)
    main_speech_embeddings = get_embeddings(main_sentences)

    # Store the embeddings in Pinecone under respective indexes
    store_embeddings_in_pinecone(interviewer_index, interviewer_sentences, interviewer_embeddings)
    store_embeddings_in_pinecone(candidate_index, candidate_sentences, candidate_embeddings)
    store_embeddings_in_pinecone(main_index, main_sentences, main_speech_embeddings)

if __name__ == "__main__":
    process_transcripts_and_store_embeddings()
