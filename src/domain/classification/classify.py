# import os
# import sys
# from src.utils.text_preprocessing import Text_Preprocessing


# class RuleBasedClassifier:
#     def __init__(
#         self,
#         keyword_file: str = r'AI002\data\top_keywords.txt',
#     ):
#         """
#         Initializes the RuleBasedClassifier

#         Args:
#             keyword_file (str): file containing keywords and their counts.
#                                 Defaults to 'data/top_keywords.txt'.
#         """
#         self.keywords = self.load_keywords_from_file(keyword_file)
#         self.preprocessor = Text_Preprocessing()

#     def load_keywords_from_file(self, filepath: str) -> dict:
#         """
#         Loads keywords from the specified file into a dictionary.

#         Args:
#             filepath (str): The path to the keywords file.

#         Returns:
#             dict: A dictionary of keywords with their counts.
#         """
#         keywords = {}
#         with open(filepath, encoding='utf-8') as f:
#             for line in f:
#                 line = line.strip()
#                 if not line:  # Skip empty lines
#                     continue
#                 parts = line.split(': ')
#                 if len(parts) == 2:
#                     word, count = parts
#                     keywords[word] = int(count)
#                 else:
#                     print(f'Warning: Invalid line: {line}')
#         return keywords

#     def classify(self, query: str) -> int:
#         processed_query = self.preprocessor(query)
#         if processed_query == "tôi chỉ hiểu tiếng việt":
#             return -1  # Special case for non-Vietnamese queries
#         for keyword in self.keywords.keys():
#             if keyword in processed_query:
#                 return 1
#         return 0

#     def __call__(self, query: str) -> int:
#         return self.classify(query)


# classify.py
import os
import sys

# Điều chỉnh đường dẫn import cho Text_Preprocessing
sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), '..', '..', 'utils',
        ),
    ),
)
from text_preprocessing import Text_Preprocessing

class RuleBasedClassifier:
    def __init__(
        self,
        keyword_file: str = r'data/top_keywords.txt', # Đường dẫn mặc định, có thể thay đổi
        stopwords_path: str = None
    ):
        """
        Initializes the RuleBasedClassifier

        Args:
            keyword_file (str): file containing keywords and their counts.
                                Defaults to 'data/top_keywords.txt'.
            stopwords_path (str): path to stopwords file.
        """
        self.keywords = self.load_keywords_from_file(keyword_file)
        if not stopwords_path:
            stopwords_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'data', 'vietnamese-stopwords-dash.txt'))
        self.preprocessor = Text_Preprocessing(stopwords_path=stopwords_path)

    def load_keywords_from_file(self, filepath: str) -> dict:
        """
        Loads keywords from the specified file into a dictionary.

        Args:
            filepath (str): The path to the keywords file.

        Returns:
            dict: A dictionary of keywords with their counts.
        """
        keywords = {}
        # Đảm bảo filepath được xử lý đúng cách (đường dẫn tuyệt đối nếu cần)
        actual_filepath = os.path.abspath(filepath)
        if not os.path.exists(actual_filepath):
            print(f"Error: Keyword file not found at {actual_filepath}. Please run extract_keyword.py first.")
            return {}
            
        with open(actual_filepath, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(': ')
                if len(parts) == 2:
                    word, count = parts
                    keywords[word] = int(count)
                else:
                    print(f'Warning: Invalid line in keyword file: {line}')
        return keywords

    def classify(self, query: str) -> int:
        processed_query = self.preprocessor(query)
        if processed_query == "tôi chỉ hiểu tiếng việt":
            return -1  # Special case for non-Vietnamese queries
        for keyword in self.keywords.keys():
            if keyword in processed_query:
                return 1
        return 0

    def __call__(self, query: str) -> int:
        return self.classify(query)