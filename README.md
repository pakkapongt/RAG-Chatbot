# RAG-Chatbot



#Workflow Summary:
User Interaction: User submits a question via the chatbot interface (bot_1.html).
Message Handling: The message is sent to the /chat endpoint via a POST request.
Text Preprocessing: The chatbot retrieves relevant context from a text file and splits it into chunks.
Retrieval: A semantic search (using embeddings and FAISS) retrieves relevant chunks from the knowledge base.
Generation: A prompt is constructed with the retrieved context and user query, which is passed to the model (GPT4All).
Answer Extraction: The model generates a response, which is processed to extract the answer and return it to the user.
Response Display: The generated answer is displayed to the user on the frontend.
