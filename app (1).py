import gradio as gr
from huggingface_hub import InferenceClient
from typing import List, Tuple
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer, util
import numpy as np
import faiss

# Initialize the Inference Client
client = InferenceClient("HuggingFaceH4/zephyr-7b-beta")

# Define the MyApp class to manage PDF processing and vector database
class MyApp:
    def __init__(self) -> None:
        self.documents = []
        self.embeddings = None
        self.index = None
        self.load_pdf("Defining_mental_health_and_mental_illness (1).pdf")  # Replace with your actual PDF file path
        self.build_vector_db()

    def load_pdf(self, file_path: str) -> None:
        """Extracts text from a PDF file and stores it in the app's documents."""
        doc = fitz.open(file_path)
        self.documents = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            self.documents.append({"page": page_num + 1, "content": text})
        print("PDF processed successfully!")

    def build_vector_db(self) -> None:
        """Builds a vector database using the content of the PDF."""
        model = SentenceTransformer('all-MiniLM-L6-v2')
        # Generate embeddings for all document contents
        self.embeddings = model.encode([doc["content"] for doc in self.documents])
        # Create a FAISS index
        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
        # Add the embeddings to the index
        self.index.add(np.array(self.embeddings))
        print("Vector database built successfully!")

    def search_documents(self, query: str, k: int = 3) -> List[str]:
        """Searches for relevant documents using vector similarity."""
        model = SentenceTransformer('all-MiniLM-L6-v2')
        # Generate an embedding for the query
        query_embedding = model.encode([query])
        # Perform a search in the FAISS index
        D, I = self.index.search(np.array(query_embedding), k)
        # Retrieve the top-k documents
        results = [self.documents[i]["content"] for i in I[0]]
        return results if results else ["No relevant documents found."]

# Instantiate the app
app = MyApp()

# Define the respond function for the chatbot
def respond(
    message: str,
    history: List[Tuple[str, str]],
    system_message: str = "You are a knowledgeable mental health support advisor.You are a supportive and empathetic mental health advisor. Remember to greet the Patient warmly, ask relevant questions, and offer supportive and insightful responses.",
    max_tokens: int = 150,
    temperature: float = 0.7,
    top_p: float = 0.9,
):
    messages = [{"role": "system", "content": system_message}]

    for val in history:
        if val[0]:
            messages.append({"role": "user", "content": val[0]})
        if val[1]:
            messages.append({"role": "assistant", "content": val[1]})

    messages.append({"role": "user", "content": message})

    # RAG - Retrieve relevant documents
    retrieved_docs = app.search_documents(message)
    context = "\n".join(retrieved_docs)
    messages.append({"role": "system", "content": "Relevant documents: " + context})

    response = ""
    for message in client.chat_completion(
        messages,
        max_tokens=max_tokens,
        stream=True,
        temperature=temperature,
        top_p=top_p,
    ):
        token = message.choices[0].delta.content
        response += token
        yield response

# Create the Gradio interface
demo = gr.Blocks()

with demo:
    gr.Markdown("🧠 **Mental health support advisor**")
    gr.Markdown(
        "‼️Disclaimer: This chatbot is based on publicly available sustainability guidelines and practices. "
        "We are not certified sustainability experts, and the use of this chatbot is at your own responsibility.‼️"
    )
    
    chatbot = gr.ChatInterface(
        respond,
        examples=[
             ["How can I manage my anxiety?"],
            ["What are some self-care tips?"],
            ["How do I deal with stress at work?"],
            ["What are some relaxation techniques?"],
            ["How can I improve my sleep?"],
            ["What should I do if I feel overwhelmed?"],
            ["How can I stay positive during tough times?"],
            ["What are some ways to practice mindfulness?"]
        ],
        title='Mental Health Advisor Health 🧠'
    )

if __name__ == "__main__":
    demo.launch()
