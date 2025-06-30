from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain.chains import RetrievalQA

# 1. Load PDF
loader = PyPDFLoader("physics.pdf")
docs = loader.load()

# 2. Split into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(docs)

# 3. Generate embeddings using sentence-transformers
embeddings = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")

# 4. Store embeddings in Chroma (local)
db = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")

# 5. Define retriever with more documents retrieved (improves answer quality)
retriever = db.as_retriever(search_kwargs={"k": 8})  # fetch top 8 chunks instead of 4

# 6. Use Ollama's local model ('llama3.2')
llm = OllamaLLM(model="llama3.2")

# 7. RAG pipeline (retrieval + answer generation)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    chain_type="stuff",   # simpler chain for small number of docs
    return_source_documents=True  # returns docs used for debugging
)

# 8. Ask a question
query = "For a number greater than 1 ,with or without decimal,trailing zeros are significant or not?"
result = qa_chain.invoke({"query": query})  # Must pass as dict

# 9. Print the answer and optionally the source documents
print("Answer:", result['result'])

print("\nSource Documents Used:")
for doc in result['source_documents']:
    print(doc.page_content[:300])  # print first 300 chars of each source
