from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import wordnet
import numpy as np
from itertools import chain
from sentence_transformers import SentenceTransformer

class DataProcessor:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.bert_model = SentenceTransformer('all-mpnet-base-v2')

    def expand_query(self, query: str) -> str:
        expanded_query = query.split()
        for word in query.split():
            synonyms = wordnet.synsets(word)
            lemmas = set(chain(*[syn.lemma_names() for syn in synonyms]))
            expanded_query.extend(list(lemmas)[:5])
        
        return ' '.join(expanded_query)

    def preprocess_texts_with_expansion(self, docs, queries):
        doc_texts = [doc['Abstract'] for doc in docs]
        query_texts = [qry['Query'] for qry in queries]
        expanded_query_texts = [self.expand_query(query) for query in query_texts]
        
        return doc_texts, expanded_query_texts

    def compute_tfidf(self, doc_texts, query_texts):
        doc_texts = [doc for doc in doc_texts if doc is not None and doc.strip() != '']
        query_texts = [qry for qry in query_texts if qry is not None and qry.strip() != '']

        combined_texts = doc_texts + query_texts
        vectors = self.vectorizer.fit_transform(combined_texts)

        doc_vectors = vectors[:len(doc_texts)]
        query_vectors = vectors[len(doc_texts):]
        
        return doc_vectors, query_vectors

    def retrieve_documents(self, doc_vectors, query_vectors):
        similarity_matrix = cosine_similarity(query_vectors, doc_vectors)
        return similarity_matrix

    def compute_embeddings_bert(self, doc_texts, query_texts):
        doc_embeddings = self.bert_model.encode(doc_texts, convert_to_tensor=True)
        query_embeddings = self.bert_model.encode(query_texts, convert_to_tensor=True)

        return doc_embeddings, query_embeddings

    def retrieve_documents_bert(self, doc_embeddings, query_embeddings):
        similarity_matrix = cosine_similarity(query_embeddings.cpu(), doc_embeddings.cpu())
        return similarity_matrix
