import os
import streamlit as st
import pickle
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.document_loaders import UnstructuredURLLoader
from dotenv import load_dotenv

# Load environment variables and configure Google API
load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Function to create a conversational QA chain
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}\n
    Question: \n{question}\n
    Answer:
    """

    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

# Streamlit interface setup
st.title("News Research Tool 📈")
st.sidebar.title("News Article URLs")

# URL input
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_google.pkl"
main_placeholder = st.empty()

if process_url_clicked:
    # Load data from URLs
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading...Started...✅✅✅")
    data = loader.load()

    # Check if data is loaded
    if not data:
        st.error("No data loaded from the URLs. Please check the URLs and try again.")
        st.stop()

    # Split data
    text_splitter = RecursiveCharacterTextSplitter(separators=['\n\n', '\n', '.', ','], chunk_size=1000)
    main_placeholder.text("Text Splitter...Started...✅✅✅")
    docs = text_splitter.split_documents(data)

    # Check if documents are available
    if not docs or not all(docs):
        st.error("Error in document splitting. No valid documents to process.")
        st.stop()

    # Create embeddings with GoogleGenerativeAIEmbeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

    # Check if embeddings can be created
    try:
        vectorstore_google = FAISS.from_documents(docs, embeddings)
    except Exception as e:
        st.error(f"Error creating embeddings: {e}")
        st.stop()

    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
    
    # Save the FAISS index
    with open(file_path, "wb") as f:
        pickle.dump(vectorstore_google, f)

loader = UnstructuredURLLoader(urls=urls)
data = loader.load()
text_splitter = RecursiveCharacterTextSplitter(separators=['\n\n', '\n', '.', ','], chunk_size=1000)
docs = text_splitter.split_documents(data)
# Handling user query
query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
            chain = get_conversational_chain()
            result = chain({"input_documents": docs, "question": query}, return_only_outputs=True)
            
            # Display the answer
            st.header("Answer")
            st.write(result["output_text"])

            # Display sources, if available (optional)
            sources = result.get("sources", "")
            if sources:
                st.subheader("Sources:")
                sources_list = sources.split("\n")
                for source in sources_list:
                    st.write(source)