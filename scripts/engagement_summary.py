import os
import logging
import warnings
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

def create_output_and_save_summary(base_path, summary_text, filename):
    """
    Creates the 'outputs' directory if it doesn't exist and saves the summary to the specified file.
    This file will be overwritten each time the function is called.
    """
    # Define the output directory path
    output_dir = os.path.join(base_path, 'outputs')

    # Check if the directory exists, if not, create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Define the path for the summary file
    summary_file_path = os.path.join(output_dir, filename)

    # Write the summary_text to the file (overwrites the file each time)
    try:
        with open(summary_file_path, 'w', encoding='utf-8') as file:
            file.write(summary_text)
        print(f"Summary successfully saved at: {summary_file_path}")
    except Exception as e:
        print(f"Error saving the summary: {e}")

    return summary_file_path  # Return the path in case it's needed later

def init_pinecone(PINECONE_API_KEY, index_dimension):
    """
    Initializes Pinecone and returns the candidate and interviewer indexes.
    """
    # Create a Pinecone instance
    pc = Pinecone(api_key=PINECONE_API_KEY)

    # Define index names
    candidate_index_name = "candidate-index"
    interviewer_index_name = "interviewer-index"

    # Check if the indexes exist, if not create them
    if candidate_index_name not in pc.list_indexes().names():
        pc.create_index(
            name=candidate_index_name,
            dimension=index_dimension,
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-west-2'
            )
        )

    if interviewer_index_name not in pc.list_indexes().names():
        pc.create_index(
            name=interviewer_index_name,
            dimension=index_dimension,
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-west-2'
            )
        )

    # Return the initialized indexes
    return pc.Index(candidate_index_name), pc.Index(interviewer_index_name)

def retrieve_relevant_segments(query, index, embedding_model, top_k=10):
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

def construct_prompt(candidate_context, interviewer_context, main_context, analysis_type):
    """
    Constructs a prompt for the Gemini model based on the analysis type.
    """
    if analysis_type == "active_listening":
        prompt = f"""You are an AI assistant specialized in analyzing candidates' active listening skills in interviews.

Your task is to evaluate the following candidate-related interview context and provide a detailed summary of their **active listening** behavior:

1. **Listening attentiveness**: Did the candidate listen attentively to the interviewer's questions, or did they seem to miss parts of the question?
2. **Interruptions**: Did the candidate interrupt the interviewer while they were speaking?
3. **Responsiveness**: Did the candidate respond appropriately to the interviewer's questions and comments?
4. **Flow of Communication**: Was the conversation smooth, or did the candidate's responses indicate a lack of attention or understanding?

**Candidate Context (Candidate's Speech):**
{candidate_context}

**Interviewer Context (Interviewer's Speech):**
{interviewer_context}

**Full Interview Context (Main Index):**
{main_context}

**Guidelines for Response:**
- Evaluate the candidate’s active listening based on the above aspects.
- Analyze whether the candidate responded appropriately and showed attentiveness.
- Comment on whether there were interruptions or if the candidate failed to follow the interviewer’s speech.
- Provide specific feedback on strengths and weaknesses.

Please ensure that your response is detailed and addresses each of the key points above.
"""
    elif analysis_type == "engagement":
        prompt = f"""You are an AI assistant specialized in analyzing candidates' engagement with interviewers.

Your task is to evaluate the following candidate-related interview context and provide a detailed summary of their **engagement** with the interviewer:

1. **Interaction Level**: Did the candidate actively interact with the interviewer, ask relevant questions, or seek clarifications?
2. **Rapport Building**: Did the candidate establish a positive rapport or connection with the interviewer?
3. **Attentiveness**: Did the candidate demonstrate attentiveness by responding thoughtfully and maintaining focus throughout the interview?

**Candidate Context (Candidate's Speech):**
{candidate_context}

**Interviewer Context (Interviewer's Speech):**
{interviewer_context}

**Full Interview Context (Main Index):**
{main_context}

**Guidelines for Response:**
- Assess the candidate's interaction level, including questions and clarifications.
- Analyze whether the candidate built a positive rapport and connection with the interviewer.
- Comment on the candidate’s attentiveness and engagement throughout the discussion.
- Provide specific feedback on strengths and areas of improvement.

Please ensure that your response is detailed and addresses each of the key points above.
"""
    return prompt

def generate_summary(query, PINECONE_API_KEY, project_base_path, gemini_model, embedding_model, embedding_dimension, analysis_type):
    """
    Generate summary based on the specified analysis type and save to outputs directory.
    """
    # Initialize Pinecone index for candidate and interviewer data
    candidate_index, interviewer_index = init_pinecone(PINECONE_API_KEY, embedding_dimension)

    # Retrieve relevant segments from the candidate and interviewer indexes
    relevant_candidate = retrieve_relevant_segments(query, candidate_index, embedding_model)
    relevant_interviewer = retrieve_relevant_segments(query, interviewer_index, embedding_model)

    # Retrieve full discussion context (main index)
    full_discussion_index = init_pinecone(PINECONE_API_KEY, embedding_dimension)[0]  # Use any index for full context
    relevant_main_context = retrieve_relevant_segments(query, full_discussion_index, embedding_model)

    # Prepare context
    candidate_context = prepare_context(relevant_candidate)
    interviewer_context = prepare_context(relevant_interviewer)
    main_context = prepare_context(relevant_main_context)

    # Construct prompt for Gemini model
    prompt = construct_prompt(candidate_context, interviewer_context, main_context, analysis_type)

    try:
        # Generate summary
        response = gemini_model.generate_content(prompt)
        summary = response.text

        # Define output filename based on analysis type
        filename = f"{analysis_type}_summary.txt"

        # Save summary to outputs directory
        summary_file_path = create_output_and_save_summary(project_base_path, summary, filename)

        print(f"{analysis_type.capitalize()} Summary saved to: {summary_file_path}")
        return summary
    except Exception as e:
        print(f"Summary generation error: {e}")
        return "Unable to generate summary."

def main():
    """
    Main execution function for generating summaries.
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
    query = "Can you evaluate the candidate's active listening skills during the interview?"

    # Generate and save active listening summary
    active_listening_summary = generate_summary(
        query=query,
        PINECONE_API_KEY=PINECONE_API_KEY,
        project_base_path=project_base_path,
        gemini_model=gemini_model,
        embedding_model=embedding_model,
        embedding_dimension=embedding_dimension,
        analysis_type="active_listening"
    )

    print("Generated Active Listening Summary:")
    print(active_listening_summary)

    # Example query for engagement analysis
    query_engagement = "Can you evaluate the candidate's engagement with the interviewer during the interview?"

    # Generate and save engagement summary
    engagement_summary = generate_summary(
        query=query_engagement,
        PINECONE_API_KEY=PINECONE_API_KEY,
        project_base_path=project_base_path,
        gemini_model=gemini_model,
        embedding_model=embedding_model,
        embedding_dimension=embedding_dimension,
        analysis_type="engagement"
    )

    print("Generated Engagement Summary:")
    print(engagement_summary)

if __name__ == "__main__":
    main()