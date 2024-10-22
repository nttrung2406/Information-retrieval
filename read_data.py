import re

def read_documents(file_path):
    with open(file_path, 'r') as file:
        content = file.read()

    documents = content.split(".I ")
    parsed_docs = []

    for doc in documents[1:]:
        doc_dict = {}
        # Document ID
        doc_id_match = re.search(r'^(\d+)', doc)
        doc_dict['ID'] = int(doc_id_match.group(1)) if doc_id_match else None
        
        # Title
        title_match = re.search(r'\.T\n(.*?)\n\.A', doc, re.DOTALL)
        doc_dict['Title'] = title_match.group(1).strip() if title_match else None
        
        # Author
        author_match = re.search(r'\.A\n(.*?)\n\.W', doc, re.DOTALL)
        doc_dict['Author'] = author_match.group(1).strip() if author_match else None
        
        # Abstract
        abstract_match = re.search(r'\.W\n(.*?)(\n\.X|\n$)', doc, re.DOTALL)
        doc_dict['Abstract'] = abstract_match.group(1).strip() if abstract_match else None
        
        # Cross-references 
        cross_refs_match = re.search(r'\.X\n(.*)', doc, re.DOTALL)
        if cross_refs_match:
            cross_refs = re.findall(r'\d+', cross_refs_match.group(1))
        else:
            cross_refs = []
        doc_dict['CrossReferences'] = cross_refs
        
        parsed_docs.append(doc_dict)
    
    return parsed_docs

def read_queries(file_path):
    with open(file_path, 'r') as file:
        content = file.read()

    queries = content.split(".I ")
    parsed_queries = []

    for qry in queries[1:]:
        qry_dict = {}
        # Query ID
        query_id_match = re.search(r'^(\d+)', qry)
        qry_dict['ID'] = int(query_id_match.group(1)) if query_id_match else None

        # Query text
        query_text_match = re.search(r'\.W\n(.*)', qry, re.DOTALL)
        qry_dict['Query'] = query_text_match.group(1).strip() if query_text_match else None

        parsed_queries.append(qry_dict)
    
    return parsed_queries

def read_mappings(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    
    query_doc_mapping = {}
    for line in lines:
        parts = line.strip().split()
        query_id = int(parts[0])  # Query ID
        doc_id = int(parts[1])    # Document ID
        
        # Query-document mappings
        if query_id not in query_doc_mapping:
            query_doc_mapping[query_id] = []
        query_doc_mapping[query_id].append(doc_id)
    
    return query_doc_mapping
