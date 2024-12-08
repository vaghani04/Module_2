import os
import logging
import warnings
from datetime import datetime
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import pinecone
from pinecone import Pinecone, ServerlessSpec
import google.generativeai as genai
from google.generativeai import GenerativeModel

# Suppress warnings
warnings.filterwarnings('ignore')
logging.getLogger('pinecone').setLevel(logging.CRITICAL)

load_dotenv()

def create_output_directory(base_path):
    """
    Create outputs directory if it doesn't exist.
    """
    output_dir = os.path.join(base_path, 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def generate_summary_filename():
    """
    Generate a unique filename for the summary based on timestamp.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"candidate_summary_{timestamp}.txt"

def init_pinecone(PINECONE_API_KEY, index_dimension):
    """
    Initializes Pinecone and returns the candidate index.
    """
    # Create a Pinecone instance
    pc = Pinecone(api_key=PINECONE_API_KEY)

    index_name = "candidate-index"

    # Check if the index exists
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=index_dimension,
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-west-2'
            )
        )
    else:
        # Validate existing index dimension
        index_info = pc.describe_index(index_name)
        if index_info.dimension != index_dimension:
            raise ValueError(f"Existing index dimension {index_info.dimension} does not match required dimension {index_dimension}")

    # Return the initialized index
    return pc.Index(index_name)

def retrieve_relevant_segments(query, index, embedding_model, top_k=5):
    """
    Retrieves the top-k relevant segments from Pinecone based on the query.
    """
    try:
        query_embedding = embedding_model.encode(query).tolist()

        result = index.query(
            vector=query_embedding,
            top_k=top_k,
            include_metadata=True
        )
        return result['matches']
    except Exception as e:
        print(f"Retrieval error: {e}")
        return []

def prepare_context(retrieved_segments):
    """
    Prepares the context for the summary generation.
    """
    context = ""
    for match in retrieved_segments:
        context += match['metadata'].get('text', '') + "\n"
    return context

def construct_prompt(context, query):
    """
    Constructs a prompt for the Gemini model.
    """
    prompt = f"""You are an AI assistant specialized in analyzing candidate performance.
    Analyze the following candidate-related interview context and provide a detailed, structured summary.

    Candidate Context:
    {context}
    
    Specific Query: {query}
    
    Guidelines for response:
    1. Highlight the candidate's key strengths.
    2. Identify potential areas of improvement.
    3. Provide specific, constructive feedback.
    4. Use clear, professional markdown formatting.
    5. If no context is available, explain the lack of information.
    """
    return prompt

def generate_interview_summary(query, PINECONE_API_KEY, project_base_path, gemini_model, embedding_model, embedding_dimension):
    """
    Generate interview summary and save to outputs/summary.txt.
    """
    # Initialize Pinecone index for candidate data
    candidate_index = init_pinecone(PINECONE_API_KEY, embedding_dimension)

    # Retrieve relevant segments from the candidate index
    relevant_candidate = retrieve_relevant_segments(query, candidate_index, embedding_model)

    # Prepare context
    context = prepare_context(relevant_candidate)

    # Construct prompt
    prompt = construct_prompt(context, query)

    try:
        # Generate summary
        response = gemini_model.generate_content(prompt)
        summary = response.text

        # Save summary to outputs/summary.txt
        output_dir = os.path.join(project_base_path, 'outputs')
        os.makedirs(output_dir, exist_ok=True)
        summary_file_path = os.path.join(output_dir, 'summary.pdf')
        with open(summary_file_path, 'w', encoding='utf-8') as f:
            f.write(summary)

        print(f"Summary saved to: {summary_file_path}")
        return summary
    except Exception as e:
        print(f"Summary generation error: {e}")
        return "Unable to generate summary."

def main():
    """
    Main execution function for interview summary generation.
    """
    # Get current script's directory
    project_base_path = os.path.dirname(os.path.abspath(__file__))
    project_base_path = os.path.dirname(project_base_path)  # Go up one level to project root

    # Configure Gemini API
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    genai.configure(api_key=GOOGLE_API_KEY)

    # Initialize Gemini models
    gemini_model = GenerativeModel("gemini-pro")

    # Load SentenceTransformer model for embeddings
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

    # Define embedding dimension
    embedding_dimension = embedding_model.get_sentence_embedding_dimension()

    # Pinecone configuration
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

    # Example query
    query = "Can you provide a detailed summary of the candidate's performance during the interview?"
    
    # Generate and save interview summary
    summary = generate_interview_summary(
        query=query, 
        PINECONE_API_KEY=PINECONE_API_KEY,
        project_base_path=project_base_path,
        gemini_model=gemini_model, 
        embedding_model=embedding_model,
        embedding_dimension=embedding_dimension
    )
    
    print("Generated Summary:")
    print(summary)

if __name__ == "__main__":
    main()
