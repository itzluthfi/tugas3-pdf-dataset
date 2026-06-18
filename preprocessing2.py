import re
from typing import Dict, List


STOPWORDS_INDONESIA = {
    "ada", "adalah", "agar", "akan", "atau", "atas", "bagaimana", "bagi",
    "bahwa", "baik", "banyak", "begini", "begitu", "belum", "bisa", "buat",
    "dalam", "dan", "dapat", "dari", "daripada", "dengan", "di", "dia",
    "ini", "itu", "jadi", "jika", "juga", "karena", "ke", "kembali",
    "kemudian", "kepada", "ketika", "lagi", "lebih", "maka", "mereka",
    "namun", "oleh", "pada", "paling", "para", "sebagai", "sebuah",
    "sebelum", "seperti", "serta", "setelah", "sudah", "supaya", "tanpa",
    "telah", "tentang", "terhadap", "tersebut", "tetapi", "tidak", "untuk",
    "yang", "yaitu", "yakni", "ia", "danau", "antara", "secara", "hingga",
}


class ManualPreprocessor:
    """Preprocessing ringan tanpa NLTK/Sastrawi.

    Tahapan dibuat eksplisit untuk tugas:
    1. case folding
    2. hapus karakter non-huruf
    3. tokenisasi regex
    4. stopword removal manual
    """

    def __init__(self, stopwords=None):
        self.stopwords = set(stopwords or STOPWORDS_INDONESIA)

    def preprocess(self, text: str) -> Dict[str, List[str]]:
        folded = text.lower()
        alpha_only = re.sub(r"[^a-zA-Z\s]", " ", folded)
        normalized_space = re.sub(r"\s+", " ", alpha_only).strip()
        tokens = re.findall(r"[a-zA-Z]+", normalized_space)
        clean_tokens = [
            token
            for token in tokens
            if len(token) > 1 and token not in self.stopwords
        ]

        return {
            "case_folded": folded,
            "normalized": normalized_space,
            "tokens": tokens,
            "clean_tokens": clean_tokens,
            "processed_text": " ".join(clean_tokens),
        }
