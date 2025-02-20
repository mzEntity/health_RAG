import utils

import pandas as pd
from llama_index.core.schema import TextNode
from llama_index.core import VectorStoreIndex


def embedding():
    file_path = 'myknowledge.xlsx'
    sheet_name = '疾病问答对'
    df = pd.read_excel(file_path, sheet_name=sheet_name, dtype=str)
    
    df['combined_text'] = "Question: " + df['question'] + "\nAnswer: " + df['answer']
    node_list = []

    for idx, row in df.iterrows():
        node = TextNode(
            id_=str(row['group']),
            text=row['combined_text'],
            metadata={
                "group": row['group'],
                "label": row['label'],
            }
        )
        node_list.append(node)
        
    index = VectorStoreIndex(node_list)
    index.storage_context.persist(persist_dir="./health_doc_emb")

if __name__ == "__main__":
    utils.setup()
    embedding()