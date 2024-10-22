from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer('all-mpnet-base-v2')

def preprocess_texts_bert(docs, queries):
    doc_texts = [doc['Abstract'] for doc in docs]
    query_texts = [qry['Query'] for qry in queries]

    return doc_texts, query_texts

def compute_embeddings_bert(doc_texts, query_texts):
    doc_embeddings = model.encode(doc_texts, convert_to_tensor=True)
    query_embeddings = model.encode(query_texts, convert_to_tensor=True)

    return doc_embeddings, query_embeddings

def retrieve_documents_bert(doc_embeddings, query_embeddings):
    similarity_matrix = cosine_similarity(query_embeddings.cpu(), doc_embeddings.cpu())

    return similarity_matrix
