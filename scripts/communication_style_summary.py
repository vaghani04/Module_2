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

# Load environment variables
load_dotenv()

import os

def create_output_and_save_summary(base_path, summary_text):
    """
    Checks if the 'outputs' directory exists. If not, creates it. 
    Then saves the communication analysis summary in a single file, overwriting previous content.
    """
    # Define the output directory path
    output_dir = os.path.join(base_path, 'outputs')

    # Check if the directory exists, if not, create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Define the path for the communication analysis summary file
    summary_file_path = os.path.join(output_dir, 'communication_analysis_summary.txt')

    # Write the summary_text to the file, overwriting any previous content
    try:
        with open(summary_file_path, 'w', encoding='utf-8') as file:
            file.write(summary_text)
        print(f"Summary successfully saved at: {summary_file_path}")
    except Exception as e:
        print(f"Error saving the summary: {e}")

    return summary_file_path  # Return the path in case it's needed later


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
    prompt = f"""You are an AI assistant specialized in analyzing candidates' communication skills in interviews.

Your task is to evaluate the following candidate-related interview context and provide a detailed summary of their communication style, focusing on **clarity, effectiveness**, and key aspects like **pace of speaking, flow, and any speech errors** (such as hesitations, fillers, or incorrect phrasing). Consider the following:

1. **Pace of Speaking**: Evaluate how fast or slow the candidate speaks. Is the pace too rapid to understand, or too slow, causing a loss of engagement? Does the pace vary appropriately with the content being conveyed?

2. **Flow of Speech**: Assess how well the candidate maintains the flow of conversation. Is there a smooth transition between ideas, or do they struggle with coherence, often jumping between topics or repeating themselves?

3. **Speech Errors**: Identify any common errors in the speech, such as hesitation, use of fillers (like "um," "uh," "you know"), incomplete sentences, or grammatical mistakes that impact the clarity and professionalism of their communication.

4. **Clarity and Effectiveness**: Evaluate how clearly the candidate conveys their ideas. Is the message well-organized, or is it difficult to follow due to poor articulation or lack of clarity? Are they able to effectively communicate complex ideas in a manner that's understandable?

**Candidate Context:**
{context}

**Specific Query:** {query}

**Guidelines for Response:**
- Assess the candidate's communication based on the above aspects.
- Provide a structured evaluation of their strengths and weaknesses.
- Offer suggestions for improvement, if applicable.
- Ensure that the response is clear, concise, and professionally structured.
- If there is insufficient information to make a full assessment, please mention that clearly.

Please ensure that your response is detailed, specific, and addresses each of the key points outlined above.
"""
    return prompt

def generate_communication_style_summary(query, PINECONE_API_KEY, project_base_path, gemini_model, embedding_model, embedding_dimension):
    """
    Generate summary of Communication Style based on clarity and effectiveness and save to outputs/communication_style_summary.txt.
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

        # Save summary to outputs/communication_style_summary.txt
        output_dir = os.path.join(project_base_path, 'outputs')
        os.makedirs(output_dir, exist_ok=True)
        summary_file_path = os.path.join(output_dir, 'communication_style_summary.txt')
        with open(summary_file_path, 'w', encoding='utf-8') as f:
            f.write(summary)

        print(f"Summary saved to: {summary_file_path}")
        return summary
    except Exception as e:
        print(f"Summary generation error: {e}")
        return "Unable to generate summary."

def main():
    """
    Main execution function for generating Communication Style summary.
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
    query = "Can you provide a detailed analysis of the candidate's communication style, focusing on their clarity and effectiveness in expression?"

    # Generate and save communication style summary
    summary = generate_communication_style_summary(
        query=query, 
        PINECONE_API_KEY=PINECONE_API_KEY,
        project_base_path=project_base_path,
        gemini_model=gemini_model, 
        embedding_model=embedding_model,
        embedding_dimension=embedding_dimension
    )
    
    print("Generated Communication Style Summary:")
    print(summary)

if __name__ == "__main__":
    main()
