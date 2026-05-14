__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import os

import chromadb
from chromadb.utils.data_loaders import ImageLoader
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from langchain_community.document_loaders import DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

from utils import ImageEmbeddings, DATA_PATH, DB_PATH


# new chromadb client
client_db = chromadb.PersistentClient(path=DB_PATH)

# create images_collection
img_collection = client_db.create_collection(
	name="imgs",
	embedding_function=ImageEmbeddings(),
	data_loader=ImageLoader()
	)

# for each category
for dir_ in os.listdir(DATA_PATH):
	dir_path = os.path.join(DATA_PATH, dir_)

	#   add images to images_collection
	img_collection.add(
	    ids=[f"{dir_}-{img_path}" for img_path in os.listdir(dir_path) if img_path.endswith('.jpg')],
	    uris=[os.path.join(dir_path, img_path) for img_path in os.listdir(dir_path) if img_path.endswith('.jpg')]
	)

	#   create documents_collection
	collection = client_db.create_collection(
	    name=f"documents_{dir_}",
	    embedding_function=OpenAIEmbeddingFunction(
	        api_key=os.getenv("OPENAI_API_KEY"),
	        model_name="text-embedding-3-small"
	    )
	)

	#   load documents
	loader = DirectoryLoader(dir_path, glob="*.txt")
	documents = loader.load()

	#   documents to chunks
	text_splitter = RecursiveCharacterTextSplitter(
	        chunk_size=300,
	        chunk_overlap=100,
	        length_function=len,
	        add_start_index=True,
	    )

	chunks = text_splitter.split_documents(documents)

	#   add chunks to documents_collection

	collection.add(
		ids=[str(j) for j in range(len(chunks))],
		documents=[chunks[j].page_content for j in range(len(chunks))],
		metadatas=[chunks[j].metadata for j in range(len(chunks))],
	)