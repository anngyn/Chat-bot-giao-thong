# AI002: Hệ thống ChatBot RAG Luật Giao Thông 🇻🇳

## 📌 Giới thiệu dự án
AI002 là một hệ thống chatbot RAG (Retrieval-Augmented Generation) chuyên biệt hỗ trợ tìm hiểu luật giao thông đường bộ Việt Nam. Hệ thống kết hợp sức mạnh của các mô hình Ngôn ngữ Lớn (LLM) hiện đại thông qua nền tảng **AWS Bedrock**, cùng công nghệ lưu trữ Database Vector tiên tiến và công cụ xử lý ngôn ngữ tự nhiên tiếng Việt chuyên sâu (**VnCoreNLP**).

---

## 🏗️ Kiến trúc Công nghệ lõi
- **Framework AI**: LlamaIndex.
- **LLM Engine**: `anthropic.claude-3-haiku-20240307-v1:0` (qua AWS Bedrock) - Đóng vai trò tổng hợp, suy luận ngữ cảnh và trả lời ngôn ngữ tự nhiên.
- **Embedding Model**: `amazon.titan-embed-text-v2:0` (qua AWS Bedrock) - Chuyển hóa văn bản luật thành vector ngữ nghĩa không gian nhiều chiều.
- **Vector Database**: ChromaDB (Lưu trữ cục bộ dạng `PersistentClient`).
- **Text Processing (NLP)**: Framework **VnCoreNLP** cho tác vụ tách từ (Word Segmentation) cốt lõi giúp tăng độ chính xác truy hồi ngữ nghĩa Tiếng Việt.
- **Giao diện Web**: Streamlit.

---

## ⚙️ Cấu trúc Module Code
Dự án được phân rã thành các module chuyên biệt (OOP) bên trong thư mục `src/`:

1. **`utils/text_preprocessing.py`**: Tiền xử lý câu hỏi, tích hợp tự động tải và khởi chạy mô hình Java `VnCoreNLP` để lấy Keyword và làm sạch rác từ vựng.
2. **`domain/classification/classify.py`**: Sử dụng thuật toán Rule-based dựa trên Term/Keyword để phân loại và từ chối các câu hỏi ngoài luồng, tối ưu chi phí hạ tầng ảo.
3. **`domain/Retrieval/database.py`**: Khai báo Bedrock Embedding và liên kết lưu trữ xuống ổ cứng với tầng Database `ChromaVectorStore`.
4. **`domain/Retrieval/retrieval.py`**: Xây dựng custom `LlamaIndex QueryEngine`, cài đặt Prompt logic tiếng Việt và kết nối trực tiếp đến AWS Bedrock Claude 3.
5. **`domain/Retrieval/chatbot.py`**: Controller tổng hợp điều hướng toàn bộ PipeLine từ lúc User đặt câu hỏi đến khi Bot render câu trả lời.
6. **`domain/main.py`**: Entry-point chứa giao diện hiển thị GUI tương tác cho Streamlit.

---

## 🚀 Hướng dẫn Cài đặt & Khởi chạy

### Bước 1: Chuẩn bị Môi trường
Yêu cầu hệ thống đã cài sẵn **Python >= 3.11**. Bạn có thể thiết lập môi trường bằng 2 cách:

#### Cách 1: Sử dụng công cụ mặc định của Python (`venv`)
```bash
# 1. Tạo môi trường ảo (Virtual Environment)
python -m venv .venv

# 2. Kích hoạt môi trường
# Với Windows PowerShell:
.\.venv\Scripts\activate
# Với MacOS/Linux:
source .venv/bin/activate

# 3. Cài đặt các thư viện phụ thuộc lõi
pip install -r requirements.txt
```

#### Cách 2: Sử dụng công cụ `uv` (Tốc độ cài đặt siêu tốc)
*Nếu máy bạn đã có môi trường cài đặt packager `uv`*
```bash
uv venv -p 3.11
# Kích hoạt môi trường (giống Cách 1)
.\.venv\Scripts\activate  # Windows
# Cài đặt siêu tốc
uv pip install -r requirements.txt
```

*(Lưu ý quan trọng: Bộ tiền xử lý `py_vncorenlp` sẽ tự động tải các gói mô hình học máy `.jar` và `.rdr` của Java ở lần chạy đầu tiên. Vậy nên máy tính của bạn bắt buộc phải có môi trường **Java/JDK (version 8 trở lên)**).*

### Bước 2: Thiết lập Biến Môi trường API
Tại thư mục gốc của dự án, bạn tạo một file tên là `.env` và điền định dạng thông tin chứng thực AWS Credentials của bạn vào. 
*(Hệ thống yêu cầu Key phải có đặc quyền truy cập gọi API model Claude và Titan từ AWS Bedrock console).*

```env
AWS_ACCESS_KEY_ID="your_aws_access_key_id"
AWS_SECRET_ACCESS_KEY="your_aws_secret_access_key"
AWS_DEFAULT_REGION="us-east-1"
```

### Bước 3: Build Tham chiếu Dữ liệu & Run Web
```bash
# Lệnh chạy ứng dụng trên giao diện Web qua Streamlit
streamlit run src/domain/main.py
```
Hệ thống sẽ host website ở địa chỉ: `http://localhost:8501`.