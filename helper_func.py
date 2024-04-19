from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from text_embeddings import  get_embedding
import numpy as np
from tqdm import tqdm

text_splitter = RecursiveCharacterTextSplitter(chunk_size=512)

def load_and_get_text(file_name, file_type = 'pdf'):
    if file_type == 'pdf':
        loader = PyPDFLoader(file_name)
    elif file_type == 'txt':
        loader = TextLoader(file_name)

    else:
        raise Exception('Invalid file type')

    docs = loader.load()
    docs = text_splitter.split_documents(docs)
    required_doc = []
    count = 0
    for doc in docs:
        if count >= 8:
            break
        required_doc.append(doc.page_content)
        count += 1
    return required_doc


def get_the_file_encoding(file_name, file_type= 'pdf'):
    texts = load_and_get_text(file_name, file_type)
    encode = []
    for text in texts:
        encode.append(get_embedding(text))
    # encode = np.concatenate(encode, axis=0)
    encode = np.mean(encode, axis=0)
    return encode / np.linalg.norm(encode, axis = 0, keepdims = True)

get_the_file_encoding('data/data/data/ACCOUNTANT/10554236.pdf')



