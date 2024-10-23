from transformers import Trainer, TrainingArguments, BertForSequenceClassification
from data_reader import DataReader
from data_processor import DataProcessor
from evaluator import Evaluator
from visualizer import Visualizer
import torch
from sklearn.model_selection import train_test_split
import numpy as np
import os

root = os.getcwd()
doc_path = os.path.join(root, 'data', 'cisi.all')
query_path = os.path.join(root, 'data', 'cisi.qry')
map_path = os.path.join(root, 'data', 'cisi.rel')

documents = DataReader.read_documents(doc_path)
queries = DataReader.read_queries(query_path)
rel_data = DataReader.read_mappings(map_path)
processor = DataProcessor()

doc_texts, expanded_query_texts = processor.preprocess_texts_with_expansion(documents, queries)
doc_vectors, query_vectors = processor.compute_tfidf(doc_texts, expanded_query_texts)

train_texts = []
train_labels = []

for query_id, relevant_docs in rel_data.items():
    for doc_id in relevant_docs:
        train_texts.append((expanded_query_texts[query_id - 1], doc_texts[doc_id - 1]))  
        train_labels.append(1)  

for query_id in rel_data.keys():
    for _ in range(len(rel_data[query_id])):
        negative_doc = np.random.choice(list(set(range(len(doc_texts))) - set(rel_data[query_id])))
        train_texts.append((expanded_query_texts[query_id - 1], doc_texts[negative_doc]))  
        train_labels.append(0)  

train_texts, val_texts, train_labels, val_labels = train_test_split(train_texts, train_labels, test_size=0.2, random_state=42)

class RetrievalDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            'input_ids': self.texts[idx][0],  
            'attention_mask': self.texts[idx][1],  
            'labels': torch.tensor(self.labels[idx], dtype=torch.long),
        }

train_dataset = RetrievalDataset(train_texts, train_labels)
val_dataset = RetrievalDataset(val_texts, val_labels)

model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    learning_rate=5e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=10,
    weight_decay=0.01,
    logging_dir='./logs',
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

trainer.train()
similarity_matrix = processor.retrieve_documents(doc_vectors, query_vectors)
avg_precision, avg_recall = Evaluator.evaluate_retrieval(similarity_matrix, rel_data)
print(f'Average Precision: {avg_precision}, Average Recall: {avg_recall}')
Visualizer.visualize_results('TF-IDF', queries, documents, similarity_matrix)

loss_values = []  
acc_values = []  
Visualizer.plot_loss_and_accuracy(loss_values, acc_values)
