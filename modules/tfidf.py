
from sklearn.feature_extraction.text import TfidfVectorizer  # pyre-ignore


class TfidfExtractor:
    def __init__(self):
        # Inisialisasi TF-IDF Vectorizer
        self.vectorizer = TfidfVectorizer()
        self.tfidf_matrix = None
        self.feature_names = []

    def fit_transform(self, documents):
        # Fit dan transform dokumen menjadi representasi TF-IDF
        self.tfidf_matrix = self.vectorizer.fit_transform(documents)

        # Simpan daftar fitur (kata-kata unik)
        self.feature_names = self.vectorizer.get_feature_names_out()

        print(f"Jumlah fitur (kata unik): {len(self.feature_names)}")
        print(f"Dimensi matriks TF-IDF : {self.tfidf_matrix.shape}")  # pyre-ignore

        return self.tfidf_matrix

    def transform_query(self, query):
        # Transform query menggunakan vectorizer yang sudah di-fit
        query_vector = self.vectorizer.transform([query])
        return query_vector

    def get_feature_names(self):
        return list(self.feature_names)

    def get_tfidf_matrix(self):
        return self.tfidf_matrix


# Testing langsung jika file ini dijalankan
if __name__ == "__main__":
    extractor = TfidfExtractor()

    # Contoh dokumen sederhana
    sample_docs = [
        "provinsi aceh terletak di ujung barat pulau sumatera",
        "provinsi bali terkenal dengan wisata pantai dan budaya",
        "provinsi jawa barat memiliki ibu kota bandung"
    ]

    print("=== Testing TF-IDF Extraction ===")
    matrix = extractor.fit_transform(sample_docs)
    print(f"\nFitur: {extractor.get_feature_names()}")

    # Test transform query
    query = "wisata pantai bali"
    query_vec = extractor.transform_query(query)
    print(f"\nQuery: '{query}'")
    print(f"Query vector shape: {query_vec.shape}")
