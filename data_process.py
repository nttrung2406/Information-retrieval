from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import wordnet  # Expand query using synonyms
from itertools import chain


def expand_query(query):
    expanded_query = query.split()  
    for word in query.split():
        synonyms = wordnet.synsets(word)  
        lemmas = set(chain(*[syn.lemma_names() for syn in synonyms]))
        expanded_query.extend(list(lemmas)[:5]) 
    
    return ' '.join(expanded_query)

def preprocess_texts_with_expansion(docs, queries):
    doc_texts = [doc['Abstract'] for doc in docs]
    query_texts = [qry['Query'] for qry in queries]

    expanded_query_texts = [expand_query(query) for query in query_texts]
    
    return doc_texts, expanded_query_texts

def compute_tfidf(doc_texts, query_texts):
    doc_texts = [doc for doc in doc_texts if doc is not None and doc.strip() != '']
    query_texts = [qry for qry in query_texts if qry is not None and qry.strip() != '']

    vectorizer = TfidfVectorizer()
    combined_texts = doc_texts + query_texts

    vectors = vectorizer.fit_transform(combined_texts)
    
    doc_vectors = vectors[:len(doc_texts)]
    query_vectors = vectors[len(doc_texts):]
    
    return doc_vectors, query_vectors, vectorizer

def retrieve_documents(doc_vectors, query_vectors):
    similarity_matrix = cosine_similarity(query_vectors, doc_vectors)
    return similarity_matrix

def compute_cosine_similarity(doc_vectors, query_vectors, query_doc_mapping):
    query_results = {}

    for query_id, query_vector in enumerate(query_vectors):
        real_query_id = query_id + 1  
        relevant_doc_ids = query_doc_mapping.get(real_query_id, [])
        if relevant_doc_ids:
            relevant_doc_vectors = np.array([doc_vectors[doc_id - 1] for doc_id in relevant_doc_ids])
            
            if len(relevant_doc_vectors) > 0:
                similarities = cosine_similarity(query_vector, relevant_doc_vectors)
                query_results[real_query_id] = {
                    'doc_ids': relevant_doc_ids,
                    'similarities': similarities.flatten().tolist()
                }
            else:
                query_results[real_query_id] = {
                    'doc_ids': relevant_doc_ids,
                    'similarities': []
                }

    return query_results
