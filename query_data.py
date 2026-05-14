__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain_core.prompts import ChatPromptTemplate
import chromadb
import cv2
from openai import OpenAI


from utils import DB_PATH, ImageEmbeddings


def classify_img(client_db, query_img):

	# TODO: error handling !
	#  currently only bgr colorspace supported
	#  what happens if no similar image found?

	collection = client_db.get_collection(name="imgs")

	embeddingFunction = ImageEmbeddings()
	img = cv2.imread(query_img)
	embeddings = embeddingFunction([img])

	results = collection.query(
		query_embeddings=embeddings,
		n_results=1
	)

	return results['ids'][0][0].split('-')[0]


def get_most_similar_chunks(client_db, query_question, img_category):
	collection = client_db.get_collection(name=f"documents_{img_category}")

	results = collection.query(
		query_texts=[query_question],
		n_results=3
	)

	return results['documents'][0], results['metadatas'][0]


def create_response(chunks_text, chunks_metadata, query_question):

	PROMPT_TEMPLATE = """
	Answer the question based only on the following context:

	{context}

	---

	Question: {question}

	"""

	prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

	context = '\n---\n'.join(chunks_text)

	prompt = prompt_template.format(context=context, question=query_question)

	model = OpenAI()
	response = model.responses.create(model="gpt-5.2", input=prompt)

	return response.output_text, chunks_metadata


if __name__ == "__main__":
	# input data: query img, query text
	query_img = '/home/phillip/Documents/current_tutorial/130_rag/code/test-1.jpg'  # path to img
	query_question = "what are those statues around the clock?"

	# create chroma db client
	client_db = chromadb.PersistentClient(path=DB_PATH)

	# classify img
	img_category = classify_img(client_db, query_img)

	# get most similar chunks
	chunks_text, chunks_metadata = get_most_similar_chunks(client_db, query_question, img_category)

	# create response
	text, sources = create_response(chunks_text, chunks_metadata, query_question)

	print(text)
	print(sources)
