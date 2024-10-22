import random
from read_data import read_documents, read_queries, read_mappings
from data_process import preprocess_texts_with_expansion, compute_tfidf, retrieve_documents
from data_process_with_bert import preprocess_texts_bert, compute_embeddings_bert, retrieve_documents_bert
from evaluate import evaluate_retrieval
import os

def visualize_results(method_name, queries, documents, similarity_matrix):
    random_query_index = random.randint(0, len(queries) - 1)
    query_id = random_query_index + 1  

    selected_query = queries[random_query_index]['Query']
    print(f"Method: {method_name}")
    print(f"Selected Query ID: {query_id}")
    print(f"Query Text: {selected_query}\n")

    retrieved_doc_indices = similarity_matrix[random_query_index].argsort()[-5:][::-1]  
    print(f"Top 5 Retrieved Documents using {method_name}:")

    for rank, doc_index in enumerate(retrieved_doc_indices):
        doc_id = doc_index + 1  
        doc_title = documents[doc_index]['Title'] if documents[doc_index]['Title'] else "No Title"
        similarity_score = similarity_matrix[random_query_index][doc_index]

        print(f"Rank {rank + 1}:")
        print(f"Document ID: {doc_id}")
        print(f"Title: {doc_title}")
        print(f"Similarity Score: {similarity_score:.4f}")
        print()

def combine_similarities(tfidf_sim, bert_sim, alpha=0.5):
    combined_sim = (alpha * tfidf_sim) + ((1 - alpha) * bert_sim)
    return combined_sim

root = os.getcwd()
doc_path = os.path.join(root, 'data', 'cisi.all')
query_path = os.path.join(root, 'data', 'cisi.qry')
map_path = os.path.join(root, 'data', 'cisi.rel')

documents = read_documents(doc_path)
queries = read_queries(query_path)
rel_data = read_mappings(map_path)

# ===================================
# TF-IDF
# ===================================
doc_texts, query_texts = preprocess_texts_with_expansion(documents, queries)
doc_vectors, query_vectors, vectorizer = compute_tfidf(doc_texts, query_texts)
similarity_matrix_tfidf = retrieve_documents(doc_vectors, query_vectors)
avg_precision_tfidf, avg_recall_tfidf = evaluate_retrieval(similarity_matrix_tfidf, rel_data)

print('TF-IDF Results:')
print(f"Average Precision: {avg_precision_tfidf:.4f}")
print(f"Average Recall: {avg_recall_tfidf:.4f}\n")
visualize_results("TF-IDF", queries, documents, similarity_matrix_tfidf)
print("\n==============================\n")

# ===================================
# BERT
# ===================================
doc_texts_bert, query_texts_bert = preprocess_texts_bert(documents, queries)
doc_embeddings, query_embeddings = compute_embeddings_bert(doc_texts_bert, query_texts_bert)
similarity_matrix_bert = retrieve_documents_bert(doc_embeddings, query_embeddings)
avg_precision_bert, avg_recall_bert = evaluate_retrieval(similarity_matrix_bert, rel_data)

print('BERT Results:')
print(f"Average Precision: {avg_precision_bert:.4f}")
print(f"Average Recall: {avg_recall_bert:.4f}\n")
visualize_results("BERT", queries, documents, similarity_matrix_bert)

# ===================================
# Compare Results
# ===================================
print("\n==============================")
print("Comparison of Methods")
print("==============================\n")
print(f"TF-IDF - Precision: {avg_precision_tfidf:.4f}, Recall: {avg_recall_tfidf:.4f}")
print(f"BERT - Precision: {avg_precision_bert:.4f}, Recall: {avg_recall_bert:.4f}")

