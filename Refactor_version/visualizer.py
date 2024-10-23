import random
import matplotlib.pyplot as plt

class Visualizer:
    @staticmethod
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

    @staticmethod
    def plot_loss_and_accuracy(loss_values, acc_values):
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(loss_values, label='Loss')
        plt.title('Training Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(acc_values, label='Accuracy', color='orange')
        plt.title('Training Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()

        plt.tight_layout()
        plt.show()
