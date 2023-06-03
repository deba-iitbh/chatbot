from langchain.vectorstores.chroma import Chroma
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import DirectoryLoader, TextLoader

seed = 43
loader = DirectoryLoader("../iitcrawl/output/", glob="**/*.txt", loader_cls=TextLoader)
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
docs = text_splitter.split_documents(documents)
persist_directory = "db"
print("Number of documents:", len(docs))

embeddings = HuggingFaceInstructEmbeddings(
    model_name="hkunlp/instructor-base", model_kwargs={"device": "cuda"}
)

vectordb = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory=persist_directory,
)
vectordb.persist()
