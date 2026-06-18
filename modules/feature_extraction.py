import math
from collections import Counter


class FeatureExtractor:
    def __init__(self):
        self.vocabulary = []
        self.tf = {}        # {filename: {term: count}}
        self.df = {}        # {term: jumlah_dokumen}
        self.idf = {}       # {term: idf_value}
        self.tfidf = {}     # {filename: {term: tfidf_value}}

    def compute_tf(self, tokens):
        return dict(Counter(tokens))

    def build_vocabulary(self):
        vocab_set = set()
        for tf_dict in self.tf.values():
            vocab_set.update(tf_dict.keys())
        self.vocabulary = sorted(vocab_set)
        return self.vocabulary

    def compute_df(self):
        for term in self.vocabulary:
            self.df[term] = sum(1 for tf in self.tf.values() if term in tf)
        return self.df

    def compute_idf(self, N):
        for term in self.vocabulary:
            if self.df[term] > 0:
                self.idf[term] = math.log10(N / self.df[term])
            else:
                self.idf[term] = 0
        return self.idf

    def compute_tfidf(self):
        for filename in self.tf:
            self.tfidf[filename] = {}
            for term, count in self.tf[filename].items():
                self.tfidf[filename][term] = count * self.idf.get(term, 0)
        return self.tfidf

    def compute_all(self, doc_tokens):
        N = len(doc_tokens)

        # Hitung TF untuk setiap dokumen
        for filename, tokens in doc_tokens.items():
            self.tf[filename] = self.compute_tf(tokens)
            print(f"  [TF] {filename}: {len(self.tf[filename])} unique terms")

        # Bangun vocabulary global
        self.build_vocabulary()
        print(f"  Vocabulary size: {len(self.vocabulary)} terms")

        # Hitung DF dan IDF
        self.compute_df()
        self.compute_idf(N)

        # Hitung TF-IDF
        self.compute_tfidf()
        print("  TF-IDF selesai dihitung.\n")

    def get_query_tfidf(self, query_tokens):
        query_tf = self.compute_tf(query_tokens)
        query_tfidf = {}
        for term, count in query_tf.items():
            if term in self.idf:
                query_tfidf[term] = count * self.idf[term]
        return query_tf, query_tfidf

    def get_vector(self, tfidf_dict):
        return [tfidf_dict.get(term, 0) for term in self.vocabulary]
