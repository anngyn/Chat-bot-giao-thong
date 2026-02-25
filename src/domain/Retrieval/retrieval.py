# # Retrieval/retrieval.py
from llama_index.core.query_engine import CustomQueryEngine
from llama_index.core.retrievers import BaseRetriever, VectorIndexRetriever
from llama_index.core.response_synthesizers import BaseSynthesizer, get_response_synthesizer
from llama_index.llms.bedrock import Bedrock
from llama_index.core import PromptTemplate
import os # Cần để lấy API key từ biến môi trường

# Prompt định nghĩa tiếng Việt
qa_prompt = PromptTemplate(
    "Bạn là trợ lý ảo giúp trả lời các câu hỏi về luật giao thông đường bộ. "
    "Tôi sẽ cung cấp cho bạn 2 thông tin: Câu hỏi và bối cảnh có chứa câu trả lời. "
    "Nhiệm vụ của bạn là tạo phản hồi dựa trên 2 thông tin đó. Lưu ý: không tự động thêm thông tin khác.\n\n"
    "Thông tin ngữ cảnh được cung cấp dưới đây.\n"
    "---------------------\n"
    "{context_str}\n"
    "---------------------\n"
    "Dựa vào thông tin ngữ cảnh trên và không sử dụng kiến thức bên ngoài, "
    "hãy trả lời câu hỏi dưới đây.\n"
    "Câu hỏi: {query_str}\n"
    "Câu trả lời (bao gồm cả trích dẫn từ tiêu đề):"
)

class RAGStringQueryEngine:
    """Query Engine dành cho RAG."""
    
    # Thay đổi type hint cho llm thành Bedrock
    def __init__(self, retriever: BaseRetriever, synthesizer: BaseSynthesizer, llm: Bedrock, qa_prompt: PromptTemplate):
        self._retriever = retriever
        self._response_synthesizer = synthesizer
        self._llm = llm
        self._qa_prompt = qa_prompt

    def custom_query(self, query_str: str) -> str:
        """Xử lý truy vấn và tạo phản hồi từ LLM."""
        nodes = self._retriever.retrieve(query_str)
        if not nodes:
            return "[Response]: Không tìm thấy thông tin liên quan."

        context_str = "\n\n".join([
            f"Tiêu đề: {node.node.metadata.get('title', 'Không có tiêu đề')}\n"
            f"Nội dung: {node.node.get_content()}"
            for node in nodes
        ])

        response = self._llm.complete(
            self._qa_prompt.format(context_str=context_str, query_str=query_str)
        )
        return response

class Retrieval:
    """Hệ thống quản lý truy vấn và trả lời."""
    
    # Cập nhật __init__ để nhận index và google_api_key
    def __init__(self, index, google_api_key: str, llm_model_name: str = "gemini-pro"):
        self.index = index

        # Cấu hình retriever
        self.retriever = VectorIndexRetriever(
            index=self.index,
            similarity_top_k=10,
        )

        # Cấu hình LLM cho RAG sử dụng Bedrock
        self.llm = Bedrock(
            model="anthropic.claude-3-haiku-20240307-v1:0",
            region_name=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY")
        )

        self.synthesizer = get_response_synthesizer(response_mode="compact")
        self.query_engine = RAGStringQueryEngine(
            retriever=self.retriever,
            synthesizer=self.synthesizer,
            llm=self.llm,
            qa_prompt=qa_prompt,
        )

    def query(self, query_str: str) -> str:
        """Thực hiện truy vấn."""
        if not self.index: # Nên có kiểm tra này nếu index có thể là None
            return "[Error]: Index chưa được tạo. Vui lòng lưu dữ liệu và tạo index trước."
        return self.query_engine.custom_query(query_str)