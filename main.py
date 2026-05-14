__import__('pysqlite3')
import sys

sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from tempfile import NamedTemporaryFile

import streamlit as st
import chromadb

from query_data import classify_img, get_most_similar_chunks, create_response
from utils import DB_PATH


client_db = chromadb.PersistentClient(path=DB_PATH)

# set title
st.title('tour-guide-ai-assistant-app')

# set header
st.header("Please upload an image")

# upload file
file = st.file_uploader("", type=["jpeg", "jpg", "png"])

if file:
    # display image
    st.image(file, use_column_width=True)


    with NamedTemporaryFile(dir='.') as f:
        f.write(file.getbuffer())
        image_path = f.name

        clf = classify_img(client_db, image_path)
        clf_ = clf.replace('_', ' ').title()

        st.write(f'You are currently looking at the **{clf_}** !\n Is there anything you would like to know about it?')

        user_question = st.text_input(f'Ask a question about **{clf_}**:')

        # write agent response
        if user_question and user_question != "":
            with st.spinner(text="In progress..."):

                chunks, metadata = get_most_similar_chunks(client_db, user_question, clf)
                response, sources = create_response(chunks, metadata, user_question)

                st.write(response)
