import argparse
from langchain_chroma import Chroma  # Aktualisierter Import
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from llama_index.core import SimpleDirectoryReader  # Ensure you're importing the correct reader
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Get OpenAI API key from environment variables
openai_api_key = os.getenv("OPENAI_API_KEY")

# Define the path for the Chroma database
CHROMA_PATH = "chroma"

# Define the prompt template for generating answers
PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""

def main():
    # Create CLI argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text

    # Load documents from the specified directory
    #documents = SimpleDirectoryReader("data/books/").load_data()  # Load your documents

    # Create an embedding function
    embedding_function = OpenAIEmbeddings()  # You can pass openai_api_key if needed

    # Create a vector database and insert the documents into it
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)
    #db.add_documents(documents)  # Add loaded documents to the database

    # Search the DB for relevant results
    results = db.similarity_search_with_relevance_scores(query_text, k=3)
    print(f"Found {len(results)} results.")
    
    # Check if results are found and relevant
    if len(results) == 0 or results[0][1] < 0.7:
        print(f"Unable to find matching results.")
        return

    # Combine context from results for the prompt
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    print(context_text)
    
    # Format the prompt with context and query
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)
    print(prompt)

    # Define OpenAI as the language model
    model = ChatOpenAI()
    response_text = model.predict(prompt)

    # Retrieve source information from results
    sources = [doc.metadata.get("source", None) for doc, _score in results]
    formatted_response = f"Response: {response_text}\nSources: {sources}"
    print(formatted_response)

if __name__ == "__main__":
    main()