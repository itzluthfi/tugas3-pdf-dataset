import csv
import hashlib
import json
import math
import os
import random
from collections import Counter
from typing import Dict, List, Tuple

from modules.loader import DocumentLoader  # pyre-ignore
from preprocessing2 import ManualPreprocessor


class ManualTfidf:
    def __init__(self):
        self.vocabulary: List[str] = []
        self.tf: Dict[str, Dict[str, int]] = {}
        self.df: Dict[str, int] = {}
        self.idf: Dict[str, float] = {}
        self.tfidf: Dict[str, Dict[str, float]] = {}

    def fit(self, documents: Dict[str, List[str]]) -> None:
        for filename, tokens in documents.items():
            self.tf[filename] = dict(Counter(tokens))

        vocab = set()
        for counts in self.tf.values():
            vocab.update(counts.keys())
        self.vocabulary = sorted(vocab)

        total_docs = len(documents)
        for term in self.vocabulary:
            self.df[term] = sum(1 for counts in self.tf.values() if term in counts)
            self.idf[term] = math.log10(total_docs / self.df[term])

        for filename, counts in self.tf.items():
            self.tfidf[filename] = {
                term: count * self.idf[term]
                for term, count in counts.items()
            }

    def transform_query(self, tokens: List[str]) -> Dict[str, float]:
        query_tf = Counter(tokens)
        return {
            term: count * self.idf[term]
            for term, count in query_tf.items()
            if term in self.idf
        }

    def vectorize(self, weights: Dict[str, float]) -> List[float]:
        return [weights.get(term, 0.0) for term in self.vocabulary]


class ManualSkipGramWord2Vec:
    def __init__(
        self,
        window_size: int = 2,
        vector_size: int = 10,
        epochs: int = 1,
        learning_rate: float = 0.025,
        max_training_pairs: int = 300,
        cache_path: str = "storage/main2_word2vec_cache.json",
        seed: int = 42,
    ):
        self.window_size = window_size
        self.vector_size = vector_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.max_training_pairs = max_training_pairs
        self.cache_path = cache_path
        self.seed = seed

        self.vocabulary: List[str] = []
        self.word_to_idx: Dict[str, int] = {}
        self.input_weights: List[List[float]] = []
        self.output_weights: List[List[float]] = []
        self.summary: Dict[str, object] = {}

    def fit(self, documents: Dict[str, List[str]]) -> None:
        cache_key = self._cache_key(documents)
        if self._load_cache(cache_key):
            print(f"  [CACHE] Word2Vec dimuat dari {self.cache_path}")
            return

        sentences = [tokens for tokens in documents.values() if tokens]
        counts = Counter(token for sentence in sentences for token in sentence)
        self.vocabulary = sorted(counts.keys())
        self.word_to_idx = {word: idx for idx, word in enumerate(self.vocabulary)}

        pairs = self._make_skipgram_pairs(sentences)
        xy_train = self._make_xy_train(pairs)
        training_pairs = pairs[:self.max_training_pairs]

        rng = random.Random(self.seed)
        self.input_weights = self._init_weights(rng)
        self.output_weights = self._init_weights(rng)
        epoch_logs = []

        for epoch in range(self.epochs):
            rng.shuffle(training_pairs)
            total_loss = 0.0
            for target_idx, context_idx in training_pairs:
                total_loss += self._train_softmax_pair(target_idx, context_idx)
            epoch_logs.append({
                "epoch": epoch + 1,
                "average_loss": round(total_loss / max(1, len(training_pairs)), 6),
                "learning_rate": self.learning_rate,
            })

        self.summary = {
            "cache_path": self.cache_path,
            "loaded_from_cache": False,
            "window_size": self.window_size,
            "vector_size": self.vector_size,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "training_architecture": "skipgram_full_softmax_manual",
            "vocabulary_size": len(self.vocabulary),
            "word_counts": self._word_count_rows(counts),
            "pair_count": len(pairs),
            "trained_pair_count": len(training_pairs),
            "x_train": [row["x_train_word"] for row in xy_train],
            "y_train": [row["y_train_word"] for row in xy_train],
            "xy_train": xy_train,
            "all_pairs": self._pair_rows(pairs),
            "layer_process": self._layer_rows(training_pairs[:20]),
            "epoch_logs": epoch_logs,
        }
        self._save_cache(cache_key)

    def document_vectors(self, documents: Dict[str, List[str]]) -> Dict[str, List[float]]:
        return {
            filename: self.sentence_vector(tokens)
            for filename, tokens in documents.items()
        }

    def sentence_vector(self, tokens: List[str]) -> List[float]:
        vectors = [
            self.input_weights[self.word_to_idx[token]]
            for token in tokens
            if token in self.word_to_idx
        ]
        if not vectors:
            return [0.0] * self.vector_size
        return [
            sum(vector[i] for vector in vectors) / len(vectors)
            for i in range(self.vector_size)
        ]

    def _make_skipgram_pairs(self, sentences: List[List[str]]) -> List[Tuple[int, int]]:
        pairs = []
        for sentence in sentences:
            indexed = [self.word_to_idx[token] for token in sentence]
            for target_pos, target_idx in enumerate(indexed):
                start = max(0, target_pos - self.window_size)
                end = min(len(indexed), target_pos + self.window_size + 1)
                for context_pos in range(start, end):
                    if context_pos != target_pos:
                        pairs.append((target_idx, indexed[context_pos]))
        return pairs

    def _make_xy_train(self, pairs: List[Tuple[int, int]]) -> List[Dict[str, object]]:
        return [
            {
                "index": idx + 1,
                "x_train_index": target_idx,
                "x_train_word": self.vocabulary[target_idx],
                "y_train_index": context_idx,
                "y_train_word": self.vocabulary[context_idx],
            }
            for idx, (target_idx, context_idx) in enumerate(pairs)
        ]

    def _train_softmax_pair(self, target_idx: int, context_idx: int) -> float:
        hidden = self.input_weights[target_idx][:]
        logits = self._output_logits(hidden)
        probabilities = self._softmax(logits)
        loss = -math.log(max(probabilities[context_idx], 1e-12))

        output_errors = probabilities[:]
        output_errors[context_idx] -= 1.0
        hidden_error = [0.0] * self.vector_size

        for word_idx in range(len(self.vocabulary)):
            for dim in range(self.vector_size):
                hidden_error[dim] += output_errors[word_idx] * self.output_weights[word_idx][dim]
                self.output_weights[word_idx][dim] -= (
                    self.learning_rate * output_errors[word_idx] * hidden[dim]
                )

        for dim in range(self.vector_size):
            self.input_weights[target_idx][dim] -= self.learning_rate * hidden_error[dim]

        return loss

    def _layer_rows(self, pairs: List[Tuple[int, int]]) -> List[Dict[str, object]]:
        rows = []
        for idx, (target_idx, context_idx) in enumerate(pairs, start=1):
            one_hot_x = self._one_hot(target_idx)
            hidden = self.input_weights[target_idx][:]
            logits = self._output_logits(hidden)
            softmax = self._softmax(logits)
            one_hot_y = self._one_hot(context_idx)
            rows.append({
                "index": idx,
                "target_word": self.vocabulary[target_idx],
                "context_word": self.vocabulary[context_idx],
                "input_one_hot": self._format_vector(one_hot_x, 0),
                "hidden_layer": self._format_vector(hidden, 4),
                "output_logits": self._format_vector(logits, 4),
                "softmax": self._format_vector(softmax, 4),
                "expected_one_hot": self._format_vector(one_hot_y, 0),
                "p_context": round(softmax[context_idx], 6),
            })
        return rows

    def _word_count_rows(self, counts: Counter) -> List[Dict[str, object]]:
        return [
            {"index": idx, "word": word, "count": counts[word]}
            for idx, word in enumerate(self.vocabulary)
        ]

    def _pair_rows(self, pairs: List[Tuple[int, int]]) -> List[Dict[str, object]]:
        return [
            {
                "index": idx + 1,
                "target_word": self.vocabulary[target_idx],
                "context_word": self.vocabulary[context_idx],
            }
            for idx, (target_idx, context_idx) in enumerate(pairs)
        ]

    def _init_weights(self, rng: random.Random) -> List[List[float]]:
        return [
            [rng.uniform(-0.01, 0.01) for _ in range(self.vector_size)]
            for _ in self.vocabulary
        ]

    def _output_logits(self, hidden: List[float]) -> List[float]:
        return [self._dot(hidden, weights) for weights in self.output_weights]

    @staticmethod
    def _softmax(values: List[float]) -> List[float]:
        max_value = max(values)
        exps = [math.exp(value - max_value) for value in values]
        total = sum(exps)
        return [value / total for value in exps]

    def _one_hot(self, active_idx: int) -> List[float]:
        return [1.0 if idx == active_idx else 0.0 for idx in range(len(self.vocabulary))]

    @staticmethod
    def _dot(vec1: List[float], vec2: List[float]) -> float:
        return sum(a * b for a, b in zip(vec1, vec2))

    @staticmethod
    def _format_vector(vector: List[float], decimals: int) -> str:
        if decimals == 0:
            return "[" + ", ".join(str(int(value)) for value in vector) + "]"
        return "[" + ", ".join(f"{value:.{decimals}f}" for value in vector) + "]"

    def _cache_key(self, documents: Dict[str, List[str]]) -> str:
        payload = {
            "params": {
                "window_size": self.window_size,
                "vector_size": self.vector_size,
                "epochs": self.epochs,
                "learning_rate": self.learning_rate,
                "max_training_pairs": self.max_training_pairs,
                "seed": self.seed,
            },
            "documents": [
                {"filename": filename, "tokens": documents[filename]}
                for filename in sorted(documents)
            ],
        }
        raw = json.dumps(payload, sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _load_cache(self, cache_key: str) -> bool:
        if not os.path.exists(self.cache_path):
            return False
        try:
            with open(self.cache_path, "r", encoding="utf-8") as file:
                cached = json.load(file)
        except (OSError, json.JSONDecodeError):
            return False
        if cached.get("cache_key") != cache_key:
            return False

        self.vocabulary = cached["vocabulary"]
        self.word_to_idx = {word: idx for idx, word in enumerate(self.vocabulary)}
        self.input_weights = cached["input_weights"]
        self.output_weights = cached["output_weights"]
        self.summary = cached["summary"]
        self.summary["loaded_from_cache"] = True
        return True

    def _save_cache(self, cache_key: str) -> None:
        os.makedirs(os.path.dirname(self.cache_path), exist_ok=True)
        payload = {
            "cache_key": cache_key,
            "vocabulary": self.vocabulary,
            "input_weights": self.input_weights,
            "output_weights": self.output_weights,
            "summary": self.summary,
        }
        with open(self.cache_path, "w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False)


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    dot = sum(a * b for a, b in zip(vec1, vec2))
    mag1 = math.sqrt(sum(a * a for a in vec1))
    mag2 = math.sqrt(sum(b * b for b in vec2))
    if mag1 == 0 or mag2 == 0:
        return 0.0
    return dot / (mag1 * mag2)


def rank_documents(query: str, tokens_by_doc: Dict[str, List[str]]) -> Dict[str, object]:
    preprocessor = ManualPreprocessor()
    query_tokens = preprocessor.preprocess(query)["clean_tokens"]

    tfidf = ManualTfidf()
    tfidf.fit(tokens_by_doc)
    query_tfidf = tfidf.transform_query(query_tokens)
    query_tfidf_vector = tfidf.vectorize(query_tfidf)

    word2vec = ManualSkipGramWord2Vec(window_size=2)
    word2vec.fit(tokens_by_doc)
    w2v_doc_vectors = word2vec.document_vectors(tokens_by_doc)
    query_w2v_vector = word2vec.sentence_vector(query_tokens)

    rows = []
    for filename in tokens_by_doc:
        tfidf_score = cosine_similarity(query_tfidf_vector, tfidf.vectorize(tfidf.tfidf[filename]))
        w2v_score = cosine_similarity(query_w2v_vector, w2v_doc_vectors[filename])
        rows.append({
            "filename": filename,
            "tfidf_score": round(tfidf_score, 6),
            "word2vec_score": round(w2v_score, 6),
        })

    return {
        "query": query,
        "query_tokens": query_tokens,
        "tfidf_vocabulary_size": len(tfidf.vocabulary),
        "tfidf_top": sorted(rows, key=lambda row: row["tfidf_score"], reverse=True),
        "word2vec_top": sorted(rows, key=lambda row: row["word2vec_score"], reverse=True),
        "word2vec_summary": word2vec.summary,
    }


def save_output(result: Dict[str, object], output_path: str) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False)


def one_hot_text(active_index: int, vocabulary_size: int) -> str:
    values = [
        "1" if index == active_index else "0"
        for index in range(vocabulary_size)
    ]
    return "[" + ",".join(values) + "]"


def save_csv_outputs(result: Dict[str, object]) -> Dict[str, str]:
    summary = result["word2vec_summary"]
    output_paths = {
        "kamus": "storage/main2_kamus.csv",
        "x_y_train": "storage/main2_x_train_y_train.csv",
        "layer": "storage/main2_layer_process.csv",
        "ranking": "storage/main2_ranking.csv",
    }
    os.makedirs("storage", exist_ok=True)

    with open(output_paths["kamus"], "w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, delimiter=";")
        writer.writerow(["index_kata", "kata", "frekuensi"])
        for row in summary["word_counts"]:
            writer.writerow([row["index"], row["word"], row["count"]])

    vocabulary_size = summary["vocabulary_size"]
    with open(output_paths["x_y_train"], "w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, delimiter=";")
        writer.writerow([
            "no",
            "x_train_index",
            "x_train_word",
            "x_train_one_hot",
            "y_train_index",
            "y_train_word",
            "y_train_one_hot",
        ])
        for row in summary["xy_train"]:
            writer.writerow([
                row["index"],
                row["x_train_index"],
                row["x_train_word"],
                one_hot_text(row["x_train_index"], vocabulary_size),
                row["y_train_index"],
                row["y_train_word"],
                one_hot_text(row["y_train_index"], vocabulary_size),
            ])

    with open(output_paths["layer"], "w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, delimiter=";")
        writer.writerow([
            "no",
            "target_word",
            "context_word",
            "input_one_hot",
            "hidden_layer",
            "output_logits",
            "softmax",
            "expected_one_hot",
            "p_context",
        ])
        for row in summary["layer_process"]:
            writer.writerow([
                row["index"],
                row["target_word"],
                row["context_word"],
                row["input_one_hot"],
                row["hidden_layer"],
                row["output_logits"],
                row["softmax"],
                row["expected_one_hot"],
                row["p_context"],
            ])

    with open(output_paths["ranking"], "w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file, delimiter=";")
        writer.writerow(["metode", "rank", "filename", "score"])
        for index, row in enumerate(result["tfidf_top"], start=1):
            writer.writerow(["tfidf", index, row["filename"], row["tfidf_score"]])
        for index, row in enumerate(result["word2vec_top"], start=1):
            writer.writerow(["word2vec_skipgram_window_2", index, row["filename"], row["word2vec_score"]])

    return output_paths


def main() -> None:
    query = input("Masukkan query: ").strip() or "wisata budaya"
    loader = DocumentLoader("docs")
    raw_documents = loader.load_all_documents()

    preprocessor = ManualPreprocessor()
    tokens_by_doc = {
        filename: preprocessor.preprocess(text)["clean_tokens"]
        for filename, text in raw_documents.items()
    }

    result = rank_documents(query, tokens_by_doc)
    save_output(result, "storage/main2_output.json")
    csv_paths = save_csv_outputs(result)

    summary = result["word2vec_summary"]
    print("\n=== RINGKASAN MANUAL TF-IDF VS WORD2VEC SKIP-GRAM ===")
    print(f"Query tokens: {result['query_tokens']}")
    print(f"Vocabulary TF-IDF: {result['tfidf_vocabulary_size']} kata")
    print(f"Vocabulary Word2Vec: {summary['vocabulary_size']} kata")
    print(f"Window Word2Vec: {summary['window_size']}")
    print(f"X_train/y_train: {len(summary['xy_train'])} pasangan")
    print(f"Cache Word2Vec: {summary['cache_path']}")
    print(f"Loaded from cache: {summary['loaded_from_cache']}")
    print("Output lengkap disimpan ke storage/main2_output.json")
    print("CSV kamus disimpan ke:", csv_paths["kamus"])
    print("CSV X_train/y_train one-hot disimpan ke:", csv_paths["x_y_train"])
    print("CSV proses layer disimpan ke:", csv_paths["layer"])
    print("CSV ranking disimpan ke:", csv_paths["ranking"])

    print("\nTop 5 TF-IDF:")
    for idx, row in enumerate(result["tfidf_top"][:5], start=1):
        print(f"{idx}. {row['filename']} | score={row['tfidf_score']}")

    print("\nTop 5 Word2Vec Skip-gram:")
    for idx, row in enumerate(result["word2vec_top"][:5], start=1):
        print(f"{idx}. {row['filename']} | score={row['word2vec_score']}")

    print("\nContoh X_train/y_train:")
    for row in summary["xy_train"][:10]:
        print(
            f"{row['index']}. X={row['x_train_word']} ({row['x_train_index']}) "
            f"-> y={row['y_train_word']} ({row['y_train_index']})"
        )

    print("\nContoh layer Word2Vec:")
    for row in summary["layer_process"][:3]:
        print(f"{row['index']}. {row['target_word']} -> {row['context_word']}")
        print(f"   input one-hot     : {row['input_one_hot'][:120]}...")
        print(f"   hidden layer      : {row['hidden_layer']}")
        print(f"   output logits     : {row['output_logits'][:120]}...")
        print(f"   softmax           : {row['softmax'][:120]}...")
        print(f"   expected one-hot  : {row['expected_one_hot'][:120]}...")


if __name__ == "__main__":
    main()
