from flask import Flask, request, render_template
import openai
from langchain.text_splitter import CharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

from langchain_community.llms import HuggingFaceHub

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import re
from gpt4all import GPT4All
from langchain_core.runnables import RunnableLambda
from langchain_core.runnables import Runnable
from langchain_core.prompt_values import ChatPromptValue
app = Flask(__name__)





llm = GPT4All(r"C:\Users\USER\.cache\gpt4all\mistral-7b-openorca.gguf2.Q4_0.gguf")


def gpt4all_runnable(prompt):
    return llm.generate(prompt)



class GPT4AllRunnable(Runnable):
    def __init__(self, llm):
        self.llm = llm

    def invoke(self, input_text, config=None):

        if isinstance(input_text, ChatPromptValue):
            input_text = input_text.to_string()  # Convert ChatPromptValue to string
        return self.llm.generate(input_text)

#Rag chatbot function
def chat_with_rag(message):
    full_text = open(r"C:\Users\USER\Downloads\rag\rag-app-main\data\doc_rag_2.txt", "r").read()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_text(full_text)

    embeddings = HuggingFaceEmbeddings()
    db = FAISS.from_texts(texts, embeddings)
    retriever = db.as_retriever()

    template = """Answer the question based only on the following context:

    {context}

    Question: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)

    model = GPT4AllRunnable(llm)

    def format_docs(docs):
        return "\n\n".join([d.page_content for d in docs])

    chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )

    return chain.invoke(message)

#Flask routes
@app.route('/')
def home():
    return render_template('Chatbot.html')

@app.route('/chat', methods=['POST'])
def chat():
    print("Chat endpoint hit!")  

    user_message = request.form.get('user_input', '')  # Safe retrieval
    print(f"User input: {user_message}")  

    if not user_message:
        print("No user input received.")  
        return {'response': "No input provided"}

    bot_message = chat_with_rag(user_message)
    print(f"Bot response before extraction: {bot_message}")  

    if not bot_message:
        print("Error: chat_with_rag() returned None")  
        return {'response': "Bot could not generate a response"}

    pattern = r"Answer:\s*(.*)"
    match = re.search(pattern, bot_message, re.DOTALL)

    if match:
        answer = match.group(1).strip()
        print("Extracted Answer:", answer)  
        return {'response': answer}
    else:
        print("Answer not found in response.")
        return {'response': "Answer not found as per context"}

if __name__ == '__main__':
    app.run()



