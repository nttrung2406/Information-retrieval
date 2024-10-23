from sklearn.metrics import precision_score, recall_score
from typing import List, Dict, Tuple

class Evaluator:
    def __init__(self, top_n: int = 5):
        self.top_n = top_n

    def evaluate_retrieval(self, similarity_matrix: List[List[float]], rel_data: Dict[int, List[int]]) -> Tuple[float, float]:
        precision_scores = []
        recall_scores = []
        
        for query_idx, similarities in enumerate(similarity_matrix):
            query_id = query_idx + 1  
            relevant_docs = rel_data.get(query_id, [])
            retrieved_docs = similarities.argsort()[-self.top_n:][::-1] + 1 
            
            relevant_binary = [1 if doc in relevant_docs else 0 for doc in retrieved_docs]
            relevant_actual = [1] * len(relevant_docs)
            
            if len(relevant_actual) > len(retrieved_docs):
                relevant_actual = relevant_actual[:len(retrieved_docs)]  # Truncate if relevant docs > retrieved docs
            elif len(retrieved_docs) > len(relevant_actual):
                relevant_actual += [0] * (len(retrieved_docs) - len(relevant_actual)) 

            if relevant_binary:
                precision = precision_score(relevant_actual, relevant_binary[:len(relevant_actual)], zero_division=0)
                recall = recall_score(relevant_actual, relevant_binary[:len(relevant_actual)], zero_division=0)
            else:
                precision, recall = 0, 0
            
            precision_scores.append(precision)
            recall_scores.append(recall)
        
        avg_precision = sum(precision_scores) / len(precision_scores) if precision_scores else 0
        avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0
        
        return avg_precision, avg_recall
