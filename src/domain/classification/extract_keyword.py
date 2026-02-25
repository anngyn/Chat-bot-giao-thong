# from __future__ import annotations

# import json
# import os
# import re
# import sys
# from collections import Counter

# import spacy
# from rake_nltk import Rake
# from sklearn.feature_extraction.text import TfidfVectorizer

# from src.utils.text_preprocessing import Text_Preprocessing

# sys.path.append(
#     os.path.abspath(
#         os.path.join(
#             os.path.dirname(__file__), '..', '..', '..',
#         ),
#     ),
# )

# # Tải stopwords tiếng Việt từ file


# def load_vietnamese_stopwords(filepath=r'data\vietnamese-stopwords-dash.txt'):
#     with open(filepath, encoding='utf-8') as f:
#         stopwords = f.read().splitlines()
#     return stopwords

# # Load dữ liệu từ file JSON


# def load_data(filepath):
#     with open(filepath, encoding='utf-8') as f:
#         data = json.load(f)
#     # Merge tất cả các content lại với nhau
#     content_list = [item['content'] for item in data]
#     return ' '.join(content_list)

# # 1. TF-IDF


# def extract_tfidf_keywords(contents, stopwords):
#     vectorizer = TfidfVectorizer(stop_words=stopwords)
#     # Truyền danh sách chứa một chuỗi duy nhất
#     X = vectorizer.fit_transform([contents])
#     feature_names = vectorizer.get_feature_names_out()

#     tfidf_keywords = {}
#     for idx in X[0].nonzero()[1]:
#         tfidf_keywords[feature_names[idx]] = X[0, idx]
#     return tfidf_keywords

# # 2. RAKE với stopwords tiếng Việt


# def extract_rake_keywords(contents, stopwords):
#     rake = Rake(stopwords=stopwords)
#     rake.extract_keywords_from_text(contents)
#     return rake.get_ranked_phrases()

# # 3. Named Entity Recognition (NER) với Spacy


# def extract_ner_keywords(contents, model='vi_core_news_sm'):
#     nlp = spacy.blank('vi')  # Bạn có thể thay thế bằng mô hình khác nếu cần
#     doc = nlp(contents)
#     entities = [(ent.text, ent.label_) for ent in doc.ents]
#     return entities


# def save_top_keywords(keywords, filename, threshold=15):
#     # Tách từ, loại bỏ từ toàn số
#     words = [
#         word for word in re.findall(
#             r'\w+', keywords,
#         ) if not word.isdigit()
#     ]

#     # Đếm tần suất các từ
#     word_count = Counter(words)

#     # Lưu vào file
#     with open(filename, 'w', encoding='utf-8') as f:
#         for word, count in word_count.items():
#             if count > threshold:
#                 f.write(f'{word}: {count}\n')


# # Đọc dữ liệu từ file
# content = load_data('data/output.json')

# # Khởi tạo đối tượng tiền xử lý
# preprocessor = Text_Preprocessing()

# # Tải stopwords tiếng Việt
# vietnamese_stopwords = load_vietnamese_stopwords()

# # Xử lý văn bản
# preprocessed_content = preprocessor(content)

# # Thực hiện trích xuất từ khóa
# tfidf_keywords = extract_tfidf_keywords(
#     preprocessed_content, vietnamese_stopwords,
# )
# rake_keywords = extract_rake_keywords(
#     preprocessed_content, vietnamese_stopwords,
# )

# # Gộp tất cả các keywords từ TF-IDF và RAKE để phân tích
# all_keywords = ' '.join(list(tfidf_keywords.keys()) + rake_keywords)

# # Sử dụng hàm để lưu các từ khóa vào file
# save_top_keywords(all_keywords, r'data\top_keywords.txt')


# extract_keyword.py
from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter

# spacy có thể cần tải mô hình 'vi_core_news_sm' trước: python -m spacy download vi_core_news_sm
import spacy 
from rake_nltk import Rake
from sklearn.feature_extraction.text import TfidfVectorizer

# Đường dẫn đến module text_preprocessing
sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), '..', '..', 'utils',
        ),
    ),
)
from text_preprocessing import Text_Preprocessing # Import trực tiếp tên module

# Tải stopwords tiếng Việt từ file
def load_vietnamese_stopwords(filepath=r'data/vietnamese-stopwords-dash.txt'):
    # Đường dẫn cần được tương đối với thư mục gốc của project hoặc là đường dẫn tuyệt đối
    # Assuming data/vietnamese-stopwords-dash.txt is relative to the project root
    # Adjust as per your project's structure
    absolute_filepath = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'vietnamese-stopwords-dash.txt'))
    with open(absolute_filepath, encoding='utf-8') as f:
        stopwords = f.read().splitlines()
    return stopwords

# Load dữ liệu từ file JSON (đã crawl)
def load_data(filepath):
    # Đảm bảo filepath trỏ đến output.json đã được crawl
    # filepath = r'data/output.json'
    absolute_filepath = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'output.json'))
    with open(absolute_filepath, encoding='utf-8') as f:
        data = json.load(f)
    
    # Merge tất cả các content lại với nhau
    content_list = [item['content'] for item in data]
    return ' '.join(content_list)

# 1. TF-IDF
def extract_tfidf_keywords(contents, stopwords):
    vectorizer = TfidfVectorizer(stop_words=stopwords)
    X = vectorizer.fit_transform([contents])
    feature_names = vectorizer.get_feature_names_out()

    tfidf_keywords = {}
    for idx in X[0].nonzero()[1]:
        tfidf_keywords[feature_names[idx]] = X[0, idx]
    return tfidf_keywords

# 2. RAKE với stopwords tiếng Việt
def extract_rake_keywords(contents, stopwords):
    rake = Rake(stopwords=stopwords)
    rake.extract_keywords_from_text(contents)
    return rake.get_ranked_phrases()


def extract_ner_keywords(contents, model='vi_core_news_sm'):
    try:
        nlp = spacy.load(model) # Tải mô hình đã được huấn luyện
        doc = nlp(contents)
        entities = [(ent.text, ent.label_) for ent in doc.ents]
        return entities
    except OSError:
        print(f"Spacy model '{model}' not found. Please run: python -m spacy download {model}")
        return []

def save_top_keywords(keywords, filename, threshold=15):
    words = [
        word for word in re.findall(
            r'\w+', keywords,
        ) if not word.isdigit()
    ]
    word_count = Counter(words)

    with open(filename, 'w', encoding='utf-8') as f:
        for word, count in word_count.items():
            if count > threshold:
                f.write(f'{word}: {count}\n')

# --- Logic chính để chạy trích xuất từ khóa ---
if __name__ == "__main__":
    # Đọc dữ liệu từ file output.json (đã được crawl)
    content = load_data(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'output.json'))

    # Khởi tạo đối tượng tiền xử lý
    # Đường dẫn stopwords cần được chỉnh sửa tương đối
    preprocessor = Text_Preprocessing(stopwords_path=os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'vietnamese-stopwords-dash.txt'))

    # Tải stopwords tiếng Việt
    vietnamese_stopwords = load_vietnamese_stopwords(filepath=os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'vietnamese-stopwords-dash.txt'))

    # Xử lý văn bản
    preprocessed_content = preprocessor(content)
    if preprocessed_content == "tôi chỉ hiểu tiếng việt":
        print("Văn bản không phải tiếng Việt hoặc quá ngắn để xử lý.")
        sys.exit()

    # Thực hiện trích xuất từ khóa
    tfidf_keywords = extract_tfidf_keywords(
        preprocessed_content, vietnamese_stopwords,
    )
    rake_keywords = extract_rake_keywords(
        preprocessed_content, vietnamese_stopwords,
    )
    

    # Gộp tất cả các keywords từ TF-IDF và RAKE để phân tích
    all_keywords = ' '.join(list(tfidf_keywords.keys()) + rake_keywords) # + [ent[0] for ent in ner_keywords] nếu dùng NER

    # Sử dụng hàm để lưu các từ khóa vào file
    # Đường dẫn top_keywords.txt
    output_keyword_file = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'top_keywords.txt')
    save_top_keywords(all_keywords, output_keyword_file)

    print(f'Top keywords successfully saved to {output_keyword_file}')