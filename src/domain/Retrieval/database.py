# from __future__ import annotations

# import json
# import os

# import chromadb
# from llama_index.core import Document
# from llama_index.core import StorageContext
# from llama_index.core import VectorStoreIndex
# from llama_index.core.node_parser import SentenceSplitter
# from llama_index.embeddings.openai import OpenAIEmbedding
# from llama_index.llms.openai import OpenAI
# from llama_index.vector_stores.chroma import ChromaVectorStore

# from llama_index.core import Settings

# class ChromaVectorStoreManager:
#     """Quản lý cơ sở dữ liệu Vector Store với Chroma."""
    
#     def __init__(self, api_key: str, collection_name: str = 'AI002', data_folder: str = 'data'):
#         self.llm = OpenAI(api_key=api_key, model='gpt-3.5-turbo-0125')
#         self.llm_embed_model = OpenAIEmbedding(api_key=api_key, model='text-embedding-ada-002')
#         Settings.llm = self.llm
#         Settings.embed_model = self.llm_embed_model
#         self.collection_name = collection_name
#         self.data_folder = data_folder
#         self.db = chromadb.PersistentClient(path=os.path.join(self.data_folder, 'chroma_db'))
#         self.collection = self.db.get_or_create_collection(self.collection_name)
#         self.index = None

#     def load_documents(self, file: str = r'data/train.json') -> list[Document]:
#         """Load documents từ file JSON."""
#         with open(file, encoding='utf-8') as f:
#             data = json.load(f)

#         documents = []
#         sentence_splitter = SentenceSplitter()

#         for item in data:
#             content = item.get('content', '')
#             title = item.get('title', '')
#             sentences = sentence_splitter.get_nodes_from_documents(
#                 [Document(text=content)]
#             )

#             for sentence in sentences:
#                 sentence.metadata = {'title': title}
#                 documents.append(Document(text=sentence.text, metadata=sentence.metadata))
        
#         print(f"Loaded and processed {len(documents)} documents.")
#         return documents

#     def store(self, documents: list[Document]) -> VectorStoreIndex:
#         """Tạo index và lưu trữ."""
#         vector_store = ChromaVectorStore(chroma_collection=self.collection)
#         storage_context = StorageContext.from_defaults(vector_store=vector_store)

#         self.index = VectorStoreIndex.from_documents(
#             documents,
#             embed_model=self.llm_embed_model,
#             storage_context=storage_context,
#             show_progress=True,
#         )
#         print("Index created and stored successfully.")
#         return self.index

#     def load_index(self) -> VectorStoreIndex:
#         """Tải index từ Vector Store."""
#         vector_store = ChromaVectorStore(chroma_collection=self.collection)
#         storage_context = StorageContext.from_defaults(vector_store=vector_store)

#         self.index = VectorStoreIndex.from_vector_store(
#             vector_store,
#             storage_context=storage_context,
#         )
#         print("Index loaded successfully.")
#         return self.index
#     def count_nodes(self) -> int:
#         """
#         Đếm số lượng nodes (dữ liệu) hiện có trong collection của Chroma.
        
#         Returns:
#             int: Số lượng nodes trong collection.
#         """
#         if not self.collection:
#             print("[Error]: Collection không tồn tại.")
#             return 0
#         return self.collection.count()
#     def delete_collection(self):
#         """
#         Xóa collection hiện tại khỏi cơ sở dữ liệu Chroma.
#         """
#         if self.collection:
#             self.db.delete_collection(self.collection_name)
#             print(f"Collection '{self.collection_name}' đã được xóa thành công.")
#         else:
#             print("[Error]: Collection không tồn tại hoặc không được khởi tạo.")

# Retrieval/database.py
from __future__ import annotations

import json
import os

import chromadb
from llama_index.core import Document
from llama_index.core import StorageContext
from llama_index.core import VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
# Thay thế GeminiEmbedding bằng BedrockEmbedding
from llama_index.embeddings.bedrock import BedrockEmbedding
from llama_index.core import Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
# Removed unused imports: FlatReader and BaseReader

# Helper function to read from a json file containing processed data
# This is a new helper function that assumes output.json contains processed chunks
def _load_data_from_json(filepath) -> list[dict]:
    with open(filepath, encoding='utf-8') as f:
        data = json.load(f)
    return data

class ChromaVectorStoreManager:
    """Quản lý cơ sở dữ liệu Vector Store với Chroma."""
    
    # Cập nhật __init__ để nhận google_api_key
    def __init__(self, google_api_key: str, collection_name: str = 'AI002', data_folder: str = 'data'):
        # Loại bỏ khởi tạo OpenAI LLM và Embedding ở đây
        # self.llm = OpenAI(api_key=api_key, model='gpt-3.5-turbo-0125') # Dòng này sẽ bị xóa
        # self.llm_embed_model = OpenAIEmbedding(api_key=api_key, model='text-embedding-ada-002') # Dòng này sẽ bị thay thế

        # Khởi tạo BedrockEmbedding
        self.embed_model = BedrockEmbedding(
            model_name="amazon.titan-embed-text-v2:0",
            region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY")
        )
        Settings.embed_model = self.embed_model
        
        # LLM sẽ không được đặt ở đây, mà ở Retrieval class
        Settings.llm = None 
        
        self.collection_name = collection_name
        self.data_folder = data_folder
        self.db_path = os.path.join(self.data_folder, 'chroma_db')
        self.client = chromadb.PersistentClient(path=self.db_path)
        self.collection = self.client.get_or_create_collection(self.collection_name)
        self.index = None

    # Thay đổi load_documents để nhận đường dẫn file JSON đã xử lý
    def load_documents(self, processed_json_file: str) -> list[Document]:
        """Load documents từ file JSON đã được tiền xử lý."""
        data = _load_data_from_json(processed_json_file) # Sử dụng helper function

        documents = []
        # Không cần SentenceSplitter ở đây nếu crawl_data.py đã tạo ra các chunk hợp lý
        # Tuy nhiên, nếu bạn muốn chia nhỏ hơn, vẫn có thể dùng SentenceSplitter
        # sentence_splitter = SentenceSplitter() 

        for item in data:
            content = item.get('content', '')
            title = item.get('title', 'Không có tiêu đề') # Đảm bảo có tiêu đề
            doc_type = item.get('type', 'Unknown') # Lấy loại tài liệu nếu có

            # Tạo Document từ mỗi mục trong JSON
            documents.append(Document(text=content, metadata={'title': title, 'type': doc_type}))
        
        print(f"Loaded and processed {len(documents)} documents from {processed_json_file}.")
        return documents

    def store(self, documents: list[Document]) -> VectorStoreIndex:
        """Tạo index và lưu trữ."""
        vector_store = ChromaVectorStore(chroma_collection=self.collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        self.index = VectorStoreIndex.from_documents(
            documents,
            embed_model=self.embed_model, # Sử dụng self.embed_model (GeminiEmbedding)
            storage_context=storage_context,
            show_progress=True,
        )
        print("Index created and stored successfully.")
        return self.index

    def load_index(self) -> VectorStoreIndex:
        """Tải index từ Vector Store."""
        vector_store = ChromaVectorStore(chroma_collection=self.collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)

        self.index = VectorStoreIndex.from_vector_store(
            vector_store,
            storage_context=storage_context,
            embed_model=self.embed_model # Cần truyền embed_model khi tải index
        )
        print("Index loaded successfully.")
        return self.index

    def count_nodes(self) -> int:
        """
        Đếm số lượng nodes (dữ liệu) hiện có trong collection của Chroma.
        """
        if not self.collection:
            print("[Error]: Collection không tồn tại.")
            return 0
        return self.collection.count()

    def delete_collection(self):
        """
        Xóa collection hiện tại khỏi cơ sở dữ liệu Chroma.
        """
        if self.collection:
            self.client.delete_collection(self.collection_name)
            print(f"Collection '{self.collection_name}' đã được xóa thành công.")
        else:
            print("[Error]: Collection không tồn tại hoặc không được khởi tạo.")