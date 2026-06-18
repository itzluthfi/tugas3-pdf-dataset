import re

try:
    from nltk.tokenize import word_tokenize  # pyre-ignore
    from nltk.corpus import stopwords  # pyre-ignore
except Exception:
    word_tokenize = None
    stopwords = None

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory  # pyre-ignore


DEFAULT_INDONESIAN_STOPWORDS = {
    'ada', 'adalah', 'agar', 'akan', 'atau', 'atas', 'bagaimana', 'bagi',
    'bahwa', 'baik', 'banyak', 'begini', 'begitu', 'belum', 'bisa', 'buat',
    'dalam', 'dan', 'dapat', 'dari', 'daripada', 'dengan', 'di', 'dia',
    'ini', 'itu', 'jadi', 'jika', 'juga', 'karena', 'ke', 'kembali',
    'kemudian', 'kepada', 'ketika', 'lagi', 'lebih', 'maka', 'mereka',
    'namun', 'oleh', 'pada', 'paling', 'para', 'sebagai', 'sebuah',
    'sebelum', 'seperti', 'serta', 'setelah', 'sudah', 'supaya', 'tanpa',
    'telah', 'tentang', 'terhadap', 'tersebut', 'tetapi', 'tidak', 'untuk',
    'yang',
}


class TextPreprocessor:
    def __init__(self):
        # Inisialisasi stopwords bahasa Indonesia
        self.stop_words = self._load_stopwords()
        # Inisialisasi stemmer Sastrawi dengan cache
        print("  Menginisialisasi Sastrawi Stemmer...")
        factory = StemmerFactory()
        self.stemmer = factory.create_stemmer()
        self._stem_cache = {}
        print("  Stemmer siap.\n")
    def _load_stopwords(self):
        if stopwords is None:
            return DEFAULT_INDONESIAN_STOPWORDS
        try:
            return set(stopwords.words('indonesian'))
        except LookupError:
            print("  NLTK stopwords tidak tersedia, memakai stopwords manual.")
            return DEFAULT_INDONESIAN_STOPWORDS
    def case_folding(self, text):
        return text.lower()
    def remove_non_alpha(self, text):
        cleaned = re.sub(r'[^a-zA-Z\s]', ' ', text)
        return re.sub(r'\s+', ' ', cleaned).strip()
    def tokenize(self, text):
        if word_tokenize is not None:
            try:
                return word_tokenize(text)
            except LookupError:
                pass
        return re.findall(r'[a-zA-Z]+', text)
    def remove_stopwords(self, tokens):
        return [t for t in tokens if t not in self.stop_words and len(t) > 1]
    def stem_word(self, word):
        if word not in self._stem_cache:
            self._stem_cache[word] = self.stemmer.stem(word)
        return self._stem_cache[word]
    def stem_tokens(self, tokens):
        return [self.stem_word(t) for t in tokens]
    def preprocess_detailed(self, text):
        original = text[:500]
        # 1. Case folding
        text = self.case_folding(text)
        # 2. Hapus karakter non-huruf
        text = self.remove_non_alpha(text)
        # 3. Tokenisasi
        tokens = self.tokenize(text)
        # 4. Stopword removal
        clean_tokens = self.remove_stopwords(tokens)
        # 5. Stemming
        stemmed_tokens = self.stem_tokens(clean_tokens)
        return {
            'original': original,
            'tokens': tokens,
            'clean_tokens': clean_tokens,
            'stemmed_tokens': stemmed_tokens,
            'processed_text': ' '.join(stemmed_tokens)
        }
