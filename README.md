# Book information retrieval

The dataset can be found in Kaggle: [https://www.kaggle.com/datasets/dmaso01dsta/cisi-a-dataset-for-information-retrieval/data]

The data were collected by the Centre for Inventions and Scientific Information ("CISI") and consist of text data about 1,460 documents and 112 associated queries. Its purpose is to be used to build models of information retrieval where a given query will return a list of document IDs relevant to the query. 

The idea is to use TF-IDF and Sentence BERT model to encode and find the relevant documents with the corresponding queries. The result returns the top 5 relevent documents. 

The benchmark is Precision and Recall for Information Retrieval.

Precision = Total number of documents retrieved that are relevant / Total number of documents that are retrieved

 Recall = Total number of documents retrieved that are relevant / Total number of relevant documents in the database
