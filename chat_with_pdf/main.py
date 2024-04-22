import streamlit as st
import os
import pickle
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
from langchain.chains.question_answering import load_qa_chain
from langchain.schema import Document

# Sidebar contents
with st.sidebar:
    st.title("Domzy  LLM Chat App ")
    st.markdown(
        '''
    ## About
    This app is an LLM-powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [HuggingFace Models](https://huggingface.co/models)
    '''
    )

def main():
    st.header("Chat with PDF 💬")

    # Upload a PDF file
    pdf = st.file_uploader("Upload your PDF", type='pdf')

    if pdf is not None:
        # Read PDF and convert to documents
        pdf_reader = PdfReader(pdf)
        text = "".join(page.extract_text() for page in pdf_reader.pages)

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        chunks = text_splitter.split_text(text)

        # Create Document objects
        documents = [Document(page_content=chunk, metadata={"source": pdf.name}) for chunk in chunks]

        # Chroma vector store
        store_name = pdf.name[:-4]
        persist_directory = f"{store_name}_chroma"
        
        if os.path.exists(persist_directory):
            # Load existing Chroma collection
            vector_store = Chroma(
                collection_name=store_name,
                embedding_function=HuggingFaceEmbeddings(),
                persist_directory=persist_directory,
            )
        else:
            # Create new Chroma collection
            embeddings = HuggingFaceEmbeddings()  
            vector_store = Chroma.from_documents(
                documents,
                embeddings,
                collection_name=store_name,
                persist_directory=persist_directory,  # Save to disk
            )

        # Set up HuggingFace LLM
        llm = HuggingFaceHub(
            huggingfacehub_api_token="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",
            repo_id="tiiuae/falcon-7b-instruct",
            # model_kwargs={"temperature": 1.0, "max_length": 256},
        )

        chain = load_qa_chain(
            llm=llm,
            chain_type="stuff",
        )
        

        # Accept user questions
        query = st.text_input("Ask questions about your PDF file:")

        if query:
            # Retrieve similar documents and run QA
            st.write(query)
            docs = vector_store.similarity_search(query=query, k=2)
            response = chain.invoke({ "input_documents": docs,"question": query})
            if "output_text" in response:
                text = response["output_text"]
                start_keyword = "Helpful Answer:"
                start_index = text.find(start_keyword)

                if start_index != -1:
                    extracted_text = text[start_index:]
                else:
                    extracted_text = "No 'Helpful Answer:' found."
                
                st.write(extracted_text)

            else:
                st.write("No 'output_text' key found in response.")
    

if __name__ is '__main__':
    main()
