import os
import urllib.request

def download_file(url, target_path):
    if not os.path.exists(target_path):
        print(f"Downloading {url} to {target_path}...")
        try:
            urllib.request.urlretrieve(url, target_path)
        except Exception as e:
            print(f"Error downloading {url}: {e}")
    else:
        print(f"File {target_path} already exists.")

def setup_vncorenlp(save_dir):
    if save_dir.endswith('/'):
        save_dir = save_dir[:-1]
    
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "models", "wordsegmenter"), exist_ok=True)
    
    # 1. Jar
    download_file(
        "https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/VnCoreNLP-1.2.jar",
        os.path.join(save_dir, "VnCoreNLP-1.2.jar")
    )
    
    # 2. Wordsegmenter models
    download_file(
        "https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/wordsegmenter/vi-vocab",
        os.path.join(save_dir, "models", "wordsegmenter", "vi-vocab")
    )
    download_file(
        "https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/wordsegmenter/wordsegmenter.rdr",
        os.path.join(save_dir, "models", "wordsegmenter", "wordsegmenter.rdr")
    )

if __name__ == "__main__":
    vncorenlp_dir = os.path.abspath(os.path.join("data", "vncorenlp"))
    setup_vncorenlp(vncorenlp_dir)
    print("VnCoreNLP setup completed at:", vncorenlp_dir)
