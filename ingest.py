import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ===============================
# PATHS
# ===============================
PDF_FOLDER = "data"
VECTORSTORE_PATH = "vectorstore"

# ===============================
# LOAD PDF FILES
# ===============================
documents = []

for file in os.listdir(PDF_FOLDER):
    if file.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join(PDF_FOLDER, file))
        documents.extend(loader.load())

print(f"✅ Loaded {len(documents)} pages")

# ===============================
# SPLIT DOCUMENTS
# ===============================
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
)

chunks = text_splitter.split_documents(documents)
print(f"✅ Created {len(chunks)} chunks")

# ===============================
# LOCAL EMBEDDINGS (FREE)
# ===============================
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ===============================
# CREATE FAISS VECTORSTORE
# ===============================
vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local(VECTORSTORE_PATH)

print("🎉 Vectorstore created successfully (NO API USED)")
