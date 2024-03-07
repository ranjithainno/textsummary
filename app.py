import streamlit as st 
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import tempfile

# Check if GPU is available and set device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

checkpoint = "facebook/bart-large-cnn"  # Using a different model with known GPU support
base_model = AutoModelForSeq2SeqLM.from_pretrained(checkpoint).to(device)
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

# File preprocessing
def file_preprocessing(file):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
        temp_file.write(file.read())
        temp_file_path = temp_file.name

    loader = PyPDFLoader(temp_file_path)  # Pass file path to PyPDFLoader
    pages = loader.load_and_split()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    texts = text_splitter.split_documents(pages)
    final_texts = ""
    for text in texts:
        final_texts += text.page_content
    return final_texts

# LLM pipeline
def llm_pipeline(input_text):
    pipe_sum = pipeline(
        'summarization',
        model=base_model,
        tokenizer=tokenizer,
        max_length=2000, 
        min_length=50)
    result = pipe_sum(input_text)
    result = result[0]['summary_text']
    return result

def main():
    st.title("PDF Summarization App using Language Model")

    uploaded_file = st.file_uploader("Upload your PDF file", type=['pdf'])

    if uploaded_file is not None:
        if st.button("Summarize"):
            input_text = file_preprocessing(uploaded_file)
            if input_text.strip():  # Check if input text is not empty
                summary = llm_pipeline(input_text)
                st.success("Summarization Complete")
                st.info(summary)
            else:
                st.error("No text found in the uploaded PDF.")

if __name__ == "__main__":
    main()
