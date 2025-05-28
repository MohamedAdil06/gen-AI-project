
!pip install langchain_community
# Install dependencies (run once)
!pip install -q langchain openai gradio

import os
from langchain.llms import OpenAI
from langchain.chains.summarize import load_summarize_chain
from langchain.docstore.document import Document
import gradio as gr

# Set your OpenAI API key here (or set in env variables)
os.environ["OPENAI_API_KEY"] = "your-openai-api-key"

# Initialize the OpenAI LLM
llm = OpenAI(temperature=0)

# Summarization chain loader - using 'map_reduce' method
chain = load_summarize_chain(llm, chain_type="map_reduce")

def summarize_text(text):
    # Convert input text to a Document object
    docs = [Document(page_content=text)]

    # Run summarization chain
    summary = chain.run(docs)

    return summary

# Gradio interface for easy web UI
iface = gr.Interface(
    fn=summarize_text,
    inputs=gr.Textbox(lines=10, label="Input Text"),
    outputs=gr.Textbox(lines=5, label="Summary"),
    title="Simple Text Summarization with LangChain + OpenAI"
)

iface.launch(share=True)
