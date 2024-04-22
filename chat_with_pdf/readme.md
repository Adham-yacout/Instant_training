# Domzy LLM Chat App

This is an LLM-powered chatbot application built using [Streamlit](https://streamlit.io/), [LangChain](https://python.langchain.com/), and [HuggingFace Models](https://huggingface.co/models). 
The application allows users to upload a PDF file and interact with it through natural language questions.The app retrieves relevant content from the PDF and answers questions using a Language Learning Model (LLM).

## Features
- Upload and read PDF files.
- Split PDF content into smaller chunks for better processing.
- Use a vector store to find relevant content from uploaded PDFs.
- Answer user questions based on PDF content using a HuggingFace LLM.
- Store vectorized representations of the PDF for later retrieval.

## Prerequisites
- Python 3.8 or later
- you will need an api key from huggingchat
- streamlit==1.18.1
- LangChain
- PyPDF2==3.0.1
- Pickle
- ChromaDB
- HuggingFace
- langchain==0.0.154
- python-dotenv==1.0.0
- faiss-cpu==1.7.4
- streamlit-extras


## Installation

Before running the app, ensure you have Python and the required libraries installed. You can install the necessary packages using `pip`:

```bash
pip install streamlit langchain PyPDF2 chromadb huggingface_hub
