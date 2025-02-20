import utils
from llama_index.core import StorageContext, load_index_from_storage
import utils
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine


class QueryManager:
    
    def __init__(self):
        self.storage_dir = "./health_doc_emb"
        self.storage_context = StorageContext.from_defaults(persist_dir=self.storage_dir)
        
        self.index = load_index_from_storage(self.storage_context)

        retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=5,
        )

        self.query_engine = RetrieverQueryEngine(
            retriever=retriever,
            response_synthesizer=None,
        )

        
    def query_one(self, question):
        response = self.query_engine.query(question)
        result = ""
        scores = []
        
        for source_node in response.source_nodes:
            result = result + source_node.node.text + "\n"
            scores.append(source_node.score)
        result = result[:-1]

        return result
            
            
if __name__ == "__main__":
    utils.setup()
    print("正在加载向量数据库...")
    queryManager = QueryManager()

    while True:
        question = input("> ")
        if question == "quit":
            break
        
        result = queryManager.query_one(question)
        
        print(question)
        print(result)