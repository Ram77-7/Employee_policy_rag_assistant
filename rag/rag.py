#Step - 1 : Import Libraries
from sentence_transformers import SentenceTransformer 
from openai import OpenAI 
from dotenv import load_dotenv
import os 
import chromadb 
from pathlib import Path
#Step - 2 : Create an .env file and store the api key in it 
# .env sits one level above the rag/ folder (next to main.py)
env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=env_path)

#Step - 3: Get the api_key from .env file and pass it to the llm
api_key = os.getenv("open_api")
if not api_key:
    raise RuntimeError(f".env not found or 'open_api' missing. Looked at: {env_path}")

llm = OpenAI(api_key=api_key)

#Step - 4: Initialize the Chroma Data Base
client = chromadb.PersistentClient(
    path = "./chroma_db"
)

collections = client.get_or_create_collection(
    name = "employee_policies"
)

#Step - 5 : Load the Embedding Model

model = SentenceTransformer('all-MiniLM-L6-v2')

#Step - 6 : Create a document
documents = [
    "Employees receive 20 days of annual leave every year.",
    "Employees must get manager approval before taking leave.",
    "Monthly salary slips are available through employee portal.",
    "Employees can request leave through HR Portal.",
    "The office parking lot opens at 8 AM every working day.",
    "Employees receive medical insurace from company."
]

print("Total documents : ", len(documents))

#Step - 7 : Convert the documents into embeddings
embeddings = model.encode(documents)

print("Number of Embeddings : ", len(embeddings))
print("Number of dimensions : ", len(embeddings[0]))

#Step - 8 : Metadata which is attached with chroma DB, to filter the data
metadatas = [
    {'department' : 'HR'},
    {'department' : 'HR'},
    {'department' : 'HR'},
    {'department' : 'HR'},
    {'department' : 'Admin'},
    {'department' : 'HR'}
]

#Step - 9 : Add the embedding, documents and metadata to chroma DB
collections.add(
    ids = ["1", "2", "3", "4", "5", "6"],
    documents = documents,
    embeddings = embeddings,
    metadatas = metadatas
)

print("Documents stored successfully!")
print("Total documents : ", collections.count())

#Step - 10 : Write a function to convert the query into embedding and then query the collections for the top 3 semantic results of the documents
def retrived_documents(query):
    query_embeddings = model.encode(query)
    results = collections.query(
        query_embeddings = query_embeddings,
        n_results = 3,
        include = ['documents', 'distances', 'metadatas']
    )

    retrived_docs = results['documents'][0]
    distances = results['distances'][0]
    for document, distance in zip(retrived_docs, distances):
        print("Document : ", document)
        print("Distance : ", distance)
    return retrived_docs 

#Step - 11 : Write a function to add context and prompt to llm, so llm answers the employee questions by using given context
def generate_answer(query, context):
    context_text = "\n".join(context) 
    prompt = f"""
    You are an employee policy assistant. 
    
    Answer the questions using supplied information using the context. 
    
    If Answer not avilable in the context. say : Sorry! I don't have enough information to answer this question.
    
    Context : 
    {context_text}
    
    Question : 
    {query}
    """
    response = llm.responses.create(
        model = 'gpt-4.1',
        input = prompt
    )

    return response.output_text


print("===================================================================================================================")
print("EMPLOYEE POLICY RAG ASSISTANT")
print("===============================================================================================================")

if __name__ == "__main__":
    # Only runs with `python rag/rag.py`, never on import
    query = input("Ask your question : ")
    context = retrived_documents(query)
    print(generate_answer(query, context))