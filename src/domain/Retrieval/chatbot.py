# from __future__ import annotations

# import sys
# import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
# from Retrieval.database import ChromaVectorStoreManager
# from Retrieval.retrieval import Retrieval
# from classfication.classify import RuleBasedClassifier

# llm_api = os.getenv("GOOGLE_API_KEY")

# class ChatBot:
#     """Hệ thống ChatBot hỗ trợ tìm kiếm thông tin luật giao thông."""
    
#     def __init__(self, api_key: str, stopwords_path: str, folder_path: str, keyword_file: str, document_file: str):
#         self.database = ChromaVectorStoreManager(api_key=api_key, data_folder=folder_path)
#         self.classifier = RuleBasedClassifier(keyword_file=keyword_file)
#         self.chat_memory_buffer = {}  # Bộ nhớ đệm câu hỏi và câu trả lời

#         # Kiểm tra số lượng nodes trong database
#         node_count = self.database.count_nodes()
#         if node_count == 0:  # Nếu chưa có dữ liệu
#             try:
#                 print("Không tìm thấy dữ liệu. Đang tạo chỉ mục mới từ tài liệu...")
#                 self.load_documents_and_store(document_file)
#                 print("Chỉ mục mới đã được tạo thành công.")
#             except Exception as e:
#                 print(f"Lỗi khi tạo chỉ mục: {e}")
#         else:  # Nếu đã có dữ liệu
#             try:
#                 print(f"Tìm thấy {node_count} nodes trong database. Đang tải chỉ mục...")
#                 self.database.load_index()
#                 print("Chỉ mục đã được tải thành công.")
#             except Exception as e:
#                 print(f"Lỗi khi tải chỉ mục: {e}")

#         self.retrieval = Retrieval(google_api_key=llm_api)
#     def load_documents_and_store(self, document_file: str):
#         """Tải tài liệu và lưu trữ."""
#         documents = self.database.load_documents(document_file)
#         self.database.store(documents)

#     def process_query(self, user_question: str) -> str:
#         if user_question in self.chat_memory_buffer:
#             return f"[Cache Hit]: {self.chat_memory_buffer[user_question]}"

#         classification_result = self.classifier(user_question)

#         if classification_result == -1:
#             return "Tôi chỉ hiểu tiếng Việt. Bạn vui lòng nhập lại nha."
#         elif classification_result == 1:
#             result = self.retrieval.query(user_question)
#             if result:
#                 self.chat_memory_buffer[user_question] = result
#                 return f" {result}"
#             else:
#                 return " Không tìm thấy thông tin liên quan."
#         else:
#             return " Câu hỏi không liên quan đến giao thông đường bộ. Bạn vui lòng hỏi câu khác nha"


# chatbot.py
from __future__ import annotations

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from Retrieval.database import ChromaVectorStoreManager
from Retrieval.retrieval import Retrieval
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'classification')))
from classification.classify import RuleBasedClassifier


class ChatBot:
    """Hệ thống ChatBot hỗ trợ tìm kiếm thông tin luật giao thông."""
    
    # Cập nhật __init__ để nhận google_api_key và processed_json_file
    def __init__(self, google_api_key: str, stopwords_path: str, folder_path: str, keyword_file: str, processed_json_file: str):
        # Truyền google_api_key cho ChromaVectorStoreManager (cho Gemini Embedding)
        self.database = ChromaVectorStoreManager(google_api_key=google_api_key, data_folder=folder_path)
        self.classifier = RuleBasedClassifier(keyword_file=keyword_file, stopwords_path=stopwords_path)
        self.chat_memory_buffer = {}  # Bộ nhớ đệm câu hỏi và câu trả lời

        # Kiểm tra số lượng nodes trong database
        node_count = self.database.count_nodes()
        if node_count == 0:  # Nếu chưa có dữ liệu
            try:
                print("Không tìm thấy dữ liệu. Đang tạo chỉ mục mới từ tài liệu...")
                # Gọi load_documents_and_store với processed_json_file
                self.load_documents_and_store(processed_json_file)
                print("Chỉ mục mới đã được tạo thành công.")
            except Exception as e:
                print(f"Lỗi khi tạo chỉ mục: {e}")
                self.index = None # Đảm bảo index là None nếu có lỗi
        else:  # Nếu đã có dữ liệu
            try:
                print(f"Tìm thấy {node_count} nodes trong database. Đang tải chỉ mục...")
                self.database.load_index() # load_index() trả về index
                print("Chỉ mục đã được tải thành công.")
            except Exception as e:
                print(f"Lỗi khi tải chỉ mục: {e}")
                self.index = None

        # Kiểm tra nếu index vẫn là None sau khi tải hoặc tạo
        if self.database.index is None:
            raise RuntimeError("Không thể khởi tạo hoặc tải chỉ mục ChromaDB. Vui lòng kiểm tra file tài liệu và cấu hình.")
        
        # Khởi tạo Retrieval với index đã được tải/tạo và google_api_key
        self.retrieval = Retrieval(index=self.database.index, google_api_key=google_api_key)

    # Cập nhật load_documents_and_store để nhận processed_json_file
    def load_documents_and_store(self, processed_json_file: str):
        """Tải tài liệu từ file JSON đã xử lý và lưu trữ vào ChromaDB."""
        documents = self.database.load_documents(processed_json_file)
        self.database.store(documents)

    def process_query(self, user_question: str) -> str:
        if user_question in self.chat_memory_buffer:
            return f"[Cache Hit]: {self.chat_memory_buffer[user_question]}"

        classification_result = self.classifier(user_question)

        if classification_result == -1:
            return "Tôi chỉ hiểu tiếng Việt. Bạn vui lòng nhập lại nha."
        elif classification_result == 1:
            # Kiểm tra self.retrieval đã được khởi tạo thành công
            if not hasattr(self, 'retrieval') or self.retrieval is None:
                return "[Error]: Hệ thống truy xuất chưa được khởi tạo đúng cách."
            
            result = self.retrieval.query(user_question)
            if result:
                self.chat_memory_buffer[user_question] = result
                return f" {result}"
            else:
                return " Không tìm thấy thông tin liên quan."
        else:
            return " Câu hỏi không liên quan đến giao thông đường bộ. Bạn vui lòng hỏi câu khác nha"