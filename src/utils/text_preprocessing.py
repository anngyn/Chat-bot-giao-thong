from __future__ import annotations

import re
import os
from langdetect import detect
import py_vncorenlp

class Text_Preprocessing():
    def __init__(
        self,
        stopwords_path=os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data','vietnamese-stopwords-dash.txt'))
    ):
        """
        Khởi tạo class, tải danh sách stopwords từ file.
        """
        with open(stopwords_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        self.stop_words = [
            line.strip()
            for line in lines
        ]
        
        # Khởi tạo VnCoreNLP
        self.vncorenlp_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'vncorenlp'))
        # Đảm bảo đã có model (sẽ tải nếu chưa có, dùng script setup_vncorenlp.py đã viết trước đó)
        import urllib.request
        if not os.path.exists(os.path.join(self.vncorenlp_dir, "VnCoreNLP-1.2.jar")):
            self._download_vncorenlp()

        self.rdrsegmenter = py_vncorenlp.VnCoreNLP(annotators=["wseg"], save_dir=self.vncorenlp_dir)

    def _download_vncorenlp(self):
        print("Đang tải VnCoreNLP framework (lần đầu tiên)...")
        os.makedirs(os.path.join(self.vncorenlp_dir, "models", "wordsegmenter"), exist_ok=True)
        
        def download_file(url, path):
            if not os.path.exists(path):
                urllib.request.urlretrieve(url, path)
                
        download_file("https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/VnCoreNLP-1.2.jar", os.path.join(self.vncorenlp_dir, "VnCoreNLP-1.2.jar"))
        download_file("https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/wordsegmenter/vi-vocab", os.path.join(self.vncorenlp_dir, "models", "wordsegmenter", "vi-vocab"))
        download_file("https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/wordsegmenter/wordsegmenter.rdr", os.path.join(self.vncorenlp_dir, "models", "wordsegmenter", "wordsegmenter.rdr"))

    def remove_stopwords(self, text):
        """
        Loại bỏ các từ dừng (stopwords) khỏi văn bản và phân tách từ với VNCoreNLP.
        """
        # Phân tách từ bằng VnCoreNLP
        sentences = self.rdrsegmenter.word_segment(text)
        
        # Gộp list 1 chiều của kết quả trả về
        words = []
        for sentence in sentences:
            words.extend(sentence.split())
            
        return ' '.join([
            w for w in words
            if w not in self.stop_words
        ])

    def lowercasing(self, text):
        """
        Chuyển văn bản thành chữ thường.
        """
        return text.lower()

    def detect_language(self, text):
        language = detect(text)
        if language != 'vi':
            return False
        return True

    def handle_character(self, text):
        # Thêm khoảng trắng giữa từ và dấu câu
        pattern = r'(\w)([^\s\w])'
        text = re.sub(pattern, r'\1 \2', text)

        pattern = r'([^\s\w])(\w)'
        text = re.sub(pattern, r'\1 \2', text)

        # Loại bỏ khoảng trắng thừa
        text = re.sub(r'\s+', ' ', text).strip()

        # Giữ lại các ký tự chữ cái, số, khoảng trắng và các dấu tiếng Việt
        result = re.sub(
            r'[^a-zA-Z0-9ÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠàáâãèéêìíòóôõùúăđĩũơưƯÇêôơư\s]',
            '', text,
        )

        return result

    def preprocess(self, text):
        return self.remove_stopwords(text.lower())

    def __call__(self, text):
        if not self.detect_language(text):
            return "tôi chỉ hiểu tiếng việt"
        return self.preprocess(text)