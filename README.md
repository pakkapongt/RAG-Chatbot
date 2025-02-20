# RAG-Chatbot


## Installation
```bash
pip install -r requirements.txt
```

# Workflow Summary:
1. User Interaction: The user submits a question via the chatbot interface (bot_1.html).
2 . Message Handling: The message is sent to the /chat endpoint via a POST request.
3. Text Preprocessing: The chatbot retrieves relevant context from a text file and splits it into chunks.
4. Retrieval: A semantic search (using embeddings and FAISS) retrieves relevant chunks from the knowledge base.
5. Generation: A prompt is constructed with the retrieved context and user query, which is passed to the model (GPT4All).
6. Answer Extraction: The model generates a response, which is processed to extract the answer and return it to the user.
7. Response Display: The generated answer is displayed to the user on the front end.
