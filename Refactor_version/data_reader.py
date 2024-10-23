import re
from typing import List, Dict

class DataReader:
    @staticmethod
    def read_documents(file_path: str) -> List[Dict]:
        with open(file_path, 'r') as file:
            content = file.read()
        
        documents = content.split(".I ")
        parsed_docs = []

        for doc in documents[1:]:
            doc_dict = {
                'ID': DataReader.extract_id(doc),
                'Title': DataReader.extract_title(doc),
                'Author': DataReader.extract_author(doc),
                'Abstract': DataReader.extract_abstract(doc),
                'CrossReferences': DataReader.extract_cross_refs(doc)
            }
            parsed_docs.append(doc_dict)
        
        return parsed_docs

    @staticmethod
    def extract_id(doc: str) -> int:
        match = re.search(r'^(\d+)', doc)
        return int(match.group(1)) if match else None

    @staticmethod
    def extract_title(doc: str) -> str:
        match = re.search(r'\.T\n(.*?)\n\.A', doc, re.DOTALL)
        return match.group(1).strip() if match else None

    @staticmethod
    def extract_author(doc: str) -> str:
        match = re.search(r'\.A\n(.*?)\n\.W', doc, re.DOTALL)
        return match.group(1).strip() if match else None

    @staticmethod
    def extract_abstract(doc: str) -> str:
        match = re.search(r'\.W\n(.*?)(\n\.X|\n$)', doc, re.DOTALL)
        return match.group(1).strip() if match else None

    @staticmethod
    def extract_cross_refs(doc: str) -> List[int]:
        match = re.search(r'\.X\n(.*)', doc, re.DOTALL)
        return [int(ref) for ref in re.findall(r'\d+', match.group(1))] if match else []
    
    @staticmethod
    def read_queries(file_path: str) -> List[Dict]:
        with open(file_path, 'r') as file:
            content = file.read()

        queries = content.split(".I ")
        parsed_queries = []

        for qry in queries[1:]:
            qry_dict = {
                'ID': DataReader.extract_query_id(qry),
                'Query': DataReader.extract_query_text(qry)
            }
            parsed_queries.append(qry_dict)
        
        return parsed_queries

    @staticmethod
    def extract_query_id(qry: str) -> int:
        match = re.search(r'^(\d+)', qry)
        return int(match.group(1)) if match else None

    @staticmethod
    def extract_query_text(qry: str) -> str:
        match = re.search(r'\.W\n(.*)', qry, re.DOTALL)
        return match.group(1).strip() if match else None

    @staticmethod
    def read_mappings(file_path: str) -> Dict[int, List[int]]:
        with open(file_path, 'r') as file:
            lines = file.readlines()
        
        query_doc_mapping = {}
        for line in lines:
            parts = line.strip().split()
            query_id = int(parts[0])
            doc_id = int(parts[1])

            if query_id not in query_doc_mapping:
                query_doc_mapping[query_id] = []
            query_doc_mapping[query_id].append(doc_id)
        
        return query_doc_mapping
