# from __future__ import annotations
# import streamlit as st
# import os
# import sys
# import warnings
# from dotenv import load_dotenv
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
# # Ignore all warnings
# warnings.filterwarnings('ignore')
# from Retrieval.chatbot import ChatBot

# # Load environment variables
# load_dotenv()

# def main():
#     """
#     Streamlit web interface for ChatBot application.
#     """
#     # Configuration
#     my_key = os.getenv('OPENAI_API_KEY')
#     if not my_key:
#         st.error("Please set your OpenAI API key in the .env file")
#         return
        
#     stopwords_path = r'data/vietnamese-stopwords-dash.txt'  # Path to stopwords file
#     folder_path = r'data'  # Path for vector store
#     keyword_file = r'data/top_keywords.txt'  # Path to keyword file for classification
#     document_file = r'data/output.json'  # Path to the JSON file containing documents

#     # Initialize ChatBot only once
#     if "chatbot" not in st.session_state:
#         st.session_state.chatbot = ChatBot(
#             api_key=my_key,
#             stopwords_path=stopwords_path,
#             folder_path=folder_path,
#             keyword_file=keyword_file,
#             document_file=document_file,
#         )

#     chatbot = st.session_state.chatbot

#     # Streamlit UI Configuration
#     st.set_page_config(page_title="LangChain Chatbot", layout="centered")
#     st.title("Viet Nam Law On Road Traffic Retrieval System")

#     # Conversation history in session state
#     if "messages" not in st.session_state:
#         st.session_state.messages = [
#             {"role": "assistant", "content": "Xin chào bạn! Mình là trợ lý hỗ trợ tìm kiếm thông tin về luật giao thông Việt Nam. Bạn cần mình giúp điều gì không?"}
#         ]

#     # Display previous messages
#     for message in st.session_state.messages:
#         with st.chat_message(message["role"]):
#             st.markdown(message["content"])

#     # Input query from the user
#     if user_input := st.chat_input("Query"):
#         # Display user message
#         st.session_state.messages.append({"role": "user", "content": user_input})
#         with st.chat_message("user"):
#             st.markdown(user_input)

#         # Generate response from chatbot
#         with st.chat_message("assistant"):
#             with st.spinner("Generating response..."):
#                 response = chatbot.process_query(user_input)
#                 st.markdown(response)
#                 st.session_state.messages.append(
#                     {"role": "assistant", "content": response}
#                 )

# if __name__ == "__main__":
#     main()

# main.py
from __future__ import annotations
import streamlit as st
import os
import sys
import warnings
from dotenv import load_dotenv

# Điều chỉnh đường dẫn sys.path cho phù hợp với cấu trúc project
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

warnings.filterwarnings('ignore') # Ignore all warnings

from Retrieval.chatbot import ChatBot

# Load environment variables
load_dotenv()

def main():
    """
    Streamlit web interface for ChatBot application.
    """
    # Configuration
    # Lấy GOOGLE_API_KEY thay vì OPENAI_API_KEY
    google_api_key = os.getenv('GOOGLE_API_KEY')
    if not google_api_key:
        st.error("Please set your GOOGLE_API_KEY in the .env file")
        return
        
    # Cấu hình đường dẫn tương đối từ thư mục gốc của project (AI002)
    # Giả định project root là thư mục chứa src, data, assets
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # Đi lên 2 cấp từ src/domain/main.py để đến thư mục gốc (AI002)
    root_dir = os.path.abspath(os.path.join(current_dir, '..', '..')) 

    stopwords_path = os.path.join(root_dir, 'data', 'vietnamese-stopwords-dash.txt')
    folder_path = os.path.join(root_dir, 'data') # Thư mục chứa chroma_db
    keyword_file = os.path.join(root_dir, 'data', 'top_keywords.txt')
    processed_json_file = os.path.join(root_dir, 'data', 'output.json') # File JSON đã xử lý từ crawl_data.py

    # Initialize ChatBot only once
    if "chatbot" not in st.session_state:
        st.session_state.chatbot = ChatBot(
            google_api_key=google_api_key, # Truyền Google API key
            stopwords_path=stopwords_path,
            folder_path=folder_path,
            keyword_file=keyword_file,
            processed_json_file=processed_json_file, # Truyền đường dẫn file JSON
        )

    chatbot = st.session_state.chatbot

    # Streamlit UI Configuration
    st.set_page_config(page_title="Chatbot Luật Giao Thông", layout="centered")
    st.title("Hệ thống truy vấn thông tin Luật Giao Thông Việt Nam")

    # Conversation history in session state
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "Xin chào bạn! Mình là trợ lý hỗ trợ tìm kiếm thông tin về luật giao thông Việt Nam. Bạn cần mình giúp điều gì không?"}
        ]

    # Display previous messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input query from the user
    if user_input := st.chat_input("Bạn muốn hỏi gì về luật giao thông?"):
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Đang tìm kiếm thông tin..."):
                response = chatbot.process_query(user_input)
                st.markdown(response)
                st.session_state.messages.append(
                    {"role": "assistant", "content": response}
                )

if __name__ == "__main__":
    main()