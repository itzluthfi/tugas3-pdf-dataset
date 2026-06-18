import hashlib
import json
import math
import os
import random
from collections import Counter
from typing import Dict, List, Tuple


class SkipGramWord2Vec:
    def __init__(
        self,
        window_size: int = 2,
        vector_size: int = 10,
        epochs: int = 1,
        learning_rate: float = 0.025,
        negative_samples: int = 0,
        min_count: int = 1,
        max_training_pairs: int = 300,
        cache_path: str = "storage/word2vec_cache.json",
        seed: int = 42,
        architecture: str = "skipgram",
        use_library: bool = True,
    ):
        self.window_size = window_size
        self.vector_size = vector_size
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.negative_samples = negative_samples
        self.min_count = min_count
        self.max_training_pairs = max_training_pairs
        self.cache_path = cache_path
        self.seed = seed
        self.architecture = architecture
        self.use_library = use_library

        self.vocabulary: List[str] = []
        self.word_to_idx: Dict[str, int] = {}
        self.input_weights: List[List[float]] = []
        self.output_weights: List[List[float]] = []
        self.training_summary = {}
        self.trained = False

    def train(self, documents: Dict[str, List[str]]) -> None:
        cache_key = self._build_cache_key(documents)
        if self._load_cache(cache_key):
            print(f"  Word2Vec ({self.architecture}) dimuat dari cache: {self.cache_path}")
            return

        sentences = [tokens for tokens in documents.values() if tokens]
        
        if self.use_library:
            from gensim.models import Word2Vec
            sg_val = 1 if self.architecture == "skipgram" else 0
            model = Word2Vec(
                sentences,
                vector_size=self.vector_size,
                window=self.window_size,
                epochs=self.epochs,
                min_count=self.min_count,
                sg=sg_val,
                seed=self.seed,
                workers=1
            )
            self.vocabulary = model.wv.index_to_key
            self.word_to_idx = {word: idx for idx, word in enumerate(self.vocabulary)}
            self.input_weights = [model.wv[word].tolist() for word in self.vocabulary]
            self.output_weights = []
            
            self.training_summary = {
                "cache_path": self.cache_path,
                "cache_key": cache_key,
                "loaded_from_cache": False,
                "window_size": self.window_size,
                "vector_size": self.vector_size,
                "epochs": self.epochs,
                "learning_rate": self.learning_rate,
                "negative_samples": self.negative_samples,
                "training_type": "gensim_library",
                "training_architecture": self.architecture,
                "use_library": True,
                "max_training_pairs": self.max_training_pairs,
                "document_count": len(sentences),
                "token_count": sum(len(sentence) for sentence in sentences),
                "vocabulary_size": len(self.vocabulary),
                "word_counts": [],
                "pair_count": 0,
                "x_train": [],
                "y_train": [],
                "xy_train": [],
                "sample_pairs": [],
                "all_pairs": [],
                "trained_pair_count": 0,
                "layer_process": [],
                "epoch_logs": [{"epoch": epoch + 1, "learning_rate": self.learning_rate, "average_loss": 0.0} for epoch in range(self.epochs)],
            }
            self.trained = True
            self._save_cache(cache_key)
            return

        word_counts = self._build_vocabulary(sentences)
        self.training_summary = {
            "cache_path": self.cache_path,
            "cache_key": cache_key,
            "loaded_from_cache": False,
            "window_size": self.window_size,
            "vector_size": self.vector_size,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "negative_samples": self.negative_samples,
            "training_type": "full_softmax",
            "training_architecture": self.architecture,
            "use_library": False,
            "max_training_pairs": self.max_training_pairs,
            "document_count": len(sentences),
            "token_count": sum(len(sentence) for sentence in sentences),
            "vocabulary_size": len(self.vocabulary),
            "word_counts": self._format_word_counts(word_counts),
            "pair_count": 0,
            "x_train": [],
            "y_train": [],
            "xy_train": [],
            "sample_pairs": [],
            "all_pairs": [],
            "trained_pair_count": 0,
            "layer_process": [],
            "epoch_logs": [],
        }

        if not self.vocabulary:
            self.trained = False
            return

        if self.architecture == "cbow":
            pairs = self._generate_cbow_pairs(sentences)
            formatted_pairs = self._format_pairs_cbow(pairs)
            xy_train = self._format_xy_train_cbow(pairs)
        else:
            pairs = self._generate_skipgram_pairs(sentences)
            formatted_pairs = self._format_pairs(pairs)
            xy_train = self._format_xy_train(pairs)

        self.training_summary["pair_count"] = len(pairs)
        self.training_summary["x_train"] = [row["x_train_word"] for row in xy_train]
        self.training_summary["y_train"] = [row["y_train_word"] for row in xy_train]
        self.training_summary["xy_train"] = xy_train
        self.training_summary["sample_pairs"] = formatted_pairs[:12]
        self.training_summary["all_pairs"] = formatted_pairs
        training_pairs = pairs[:self.max_training_pairs]
        self.training_summary["trained_pair_count"] = len(training_pairs)
        if not pairs:
            self.trained = False
            return

        rng = random.Random(self.seed)
        vocab_size = len(self.vocabulary)
        self.input_weights = self._initialize_weights(rng, vocab_size)
        self.output_weights = self._initialize_weights(rng, vocab_size)

        for epoch in range(self.epochs):
            rng.shuffle(training_pairs)
            alpha = self.learning_rate * (1 - (epoch / max(1, self.epochs)))
            alpha = max(alpha, self.learning_rate * 0.1)
            epoch_loss = 0.0
            update_count = 0

            if self.architecture == "cbow":
                for context_indices, target_idx in training_pairs:
                    epoch_loss += self._update_softmax_pair_cbow(context_indices, target_idx, alpha)
                    update_count += 1
            else:
                for target_idx, context_idx in training_pairs:
                    epoch_loss += self._update_softmax_pair(target_idx, context_idx, alpha)
                    update_count += 1

            if self._should_log_epoch(epoch):
                average_loss = epoch_loss / max(1, update_count)
                self.training_summary["epoch_logs"].append({
                    "epoch": epoch + 1,
                    "learning_rate": round(alpha, 6),
                    "average_loss": round(average_loss, 6),
                })

        self.trained = True
        self.training_summary["trained"] = self.trained
        if self.architecture == "cbow":
            self.training_summary["layer_process"] = self._build_layer_process_cbow(training_pairs[:20])
        else:
            self.training_summary["layer_process"] = self._build_layer_process(training_pairs[:20])
        self._save_cache(cache_key)

    def get_document_vectors(self, documents: Dict[str, List[str]], doc_weights: Dict[str, Dict[str, float]] = None) -> Dict[str, List[float]]:
        return {
            filename: self.get_sentence_vector(
                tokens,
                weights=doc_weights.get(filename) if doc_weights is not None else None
            )
            for filename, tokens in documents.items()
        }

    def get_sentence_vector(self, tokens: List[str], weights: Dict[str, float] = None) -> List[float]:
        vectors = []
        token_weights = []
        for token in tokens:
            if token in self.word_to_idx:
                vectors.append(self.input_weights[self.word_to_idx[token]])
                if weights is not None:
                    token_weights.append(max(weights.get(token, 0.0), 1e-5))
                else:
                    token_weights.append(1.0)

        if not vectors:
            return [0.0] * self.vector_size

        total_weight = sum(token_weights)
        return [
            sum(vectors[i][dimension] * token_weights[i] for i in range(len(vectors))) / total_weight
            for dimension in range(self.vector_size)
        ]

    def _build_vocabulary(self, sentences: List[List[str]]) -> Counter:
        counts = Counter(token for sentence in sentences for token in sentence)
        self.vocabulary = sorted(
            token for token, count in counts.items() if count >= self.min_count
        )
        self.word_to_idx = {word: idx for idx, word in enumerate(self.vocabulary)}
        return counts

    def _generate_skipgram_pairs(self, sentences: List[List[str]]) -> List[Tuple[int, int]]:
        pairs = []

        for sentence in sentences:
            indexed_sentence = [
                self.word_to_idx[token]
                for token in sentence
                if token in self.word_to_idx
            ]

            for target_pos, target_idx in enumerate(indexed_sentence):
                start = max(0, target_pos - self.window_size)
                end = min(len(indexed_sentence), target_pos + self.window_size + 1)

                for context_pos in range(start, end):
                    if context_pos != target_pos:
                        pairs.append((target_idx, indexed_sentence[context_pos]))

        return pairs

    def _generate_cbow_pairs(self, sentences: List[List[str]]) -> List[Tuple[List[int], int]]:
        pairs = []

        for sentence in sentences:
            indexed_sentence = [
                self.word_to_idx[token]
                for token in sentence
                if token in self.word_to_idx
            ]

            for target_pos, target_idx in enumerate(indexed_sentence):
                context_indices = []
                start = max(0, target_pos - self.window_size)
                end = min(len(indexed_sentence), target_pos + self.window_size + 1)

                for context_pos in range(start, end):
                    if context_pos != target_pos:
                        context_indices.append(indexed_sentence[context_pos])

                if context_indices:
                    pairs.append((context_indices, target_idx))

        return pairs

    def _format_word_counts(self, word_counts: Counter) -> List[Dict[str, int]]:
        return [
            {
                "index": self.word_to_idx[word] + 1,
                "word": word,
                "count": word_counts[word],
            }
            for word in self.vocabulary
        ]

    def _format_pairs(self, pairs: List[Tuple[int, int]]) -> List[Dict[str, str]]:
        idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        return [
            {
                "index": idx + 1,
                "target": idx_to_word[target_idx],
                "context": idx_to_word[context_idx],
            }
            for idx, (target_idx, context_idx) in enumerate(pairs)
        ]

    def _format_pairs_cbow(self, pairs: List[Tuple[List[int], int]]) -> List[Dict[str, str]]:
        idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        return [
            {
                "index": idx + 1,
                "target": idx_to_word[target_idx],
                "context": ", ".join(idx_to_word[c_idx] for c_idx in context_indices),
            }
            for idx, (context_indices, target_idx) in enumerate(pairs)
        ]

    def _format_xy_train(self, pairs: List[Tuple[int, int]]) -> List[Dict[str, object]]:
        idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        return [
            {
                "index": idx + 1,
                "x_train_index": target_idx,
                "x_train_word": idx_to_word[target_idx],
                "y_train_index": context_idx,
                "y_train_word": idx_to_word[context_idx],
            }
            for idx, (target_idx, context_idx) in enumerate(pairs)
        ]

    def _format_xy_train_cbow(self, pairs: List[Tuple[List[int], int]]) -> List[Dict[str, object]]:
        idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        return [
            {
                "index": idx + 1,
                "x_train_index": context_indices,
                "x_train_word": ", ".join(idx_to_word[c_idx] for c_idx in context_indices),
                "y_train_index": target_idx,
                "y_train_word": idx_to_word[target_idx],
            }
            for idx, (context_indices, target_idx) in enumerate(pairs)
        ]

    def _should_log_epoch(self, epoch: int) -> bool:
        return True

    def _initialize_weights(self, rng: random.Random, vocab_size: int) -> List[List[float]]:
        return [
            [rng.uniform(-0.01, 0.01) for _ in range(self.vector_size)]
            for _ in range(vocab_size)
        ]

    def _update_softmax_pair(self, target_idx: int, context_idx: int, alpha: float) -> float:
        hidden_vector = self.input_weights[target_idx][:]
        output_scores = self._compute_output_scores(hidden_vector)
        probabilities = self._softmax(output_scores)
        loss = -math.log(max(probabilities[context_idx], 1e-12))

        output_errors = probabilities[:]
        output_errors[context_idx] -= 1.0
        hidden_error = [0.0] * self.vector_size

        for word_idx in range(len(self.vocabulary)):
            for dimension in range(self.vector_size):
                hidden_error[dimension] += output_errors[word_idx] * self.output_weights[word_idx][dimension]
                self.output_weights[word_idx][dimension] -= alpha * output_errors[word_idx] * hidden_vector[dimension]

        for dimension in range(self.vector_size):
            self.input_weights[target_idx][dimension] -= alpha * hidden_error[dimension]

        return loss

    def _update_softmax_pair_cbow(self, context_indices: List[int], target_idx: int, alpha: float) -> float:
        C = len(context_indices)
        hidden_vector = [0.0] * self.vector_size
        for idx in context_indices:
            for d in range(self.vector_size):
                hidden_vector[d] += self.input_weights[idx][d]
        for d in range(self.vector_size):
            hidden_vector[d] /= C

        output_scores = self._compute_output_scores(hidden_vector)
        probabilities = self._softmax(output_scores)
        loss = -math.log(max(probabilities[target_idx], 1e-12))

        output_errors = probabilities[:]
        output_errors[target_idx] -= 1.0
        hidden_error = [0.0] * self.vector_size

        for word_idx in range(len(self.vocabulary)):
            for dimension in range(self.vector_size):
                hidden_error[dimension] += output_errors[word_idx] * self.output_weights[word_idx][dimension]
                self.output_weights[word_idx][dimension] -= alpha * output_errors[word_idx] * hidden_vector[dimension]

        for idx in context_indices:
            for dimension in range(self.vector_size):
                self.input_weights[idx][dimension] -= (alpha * hidden_error[dimension]) / C

        return loss

    def _build_layer_process(self, pairs: List[Tuple[int, int]]) -> List[Dict[str, object]]:
        layer_rows = []

        for row_idx, (target_idx, context_idx) in enumerate(pairs, start=1):
            hidden_vector = self.input_weights[target_idx][:]
            output_scores = self._compute_output_scores(hidden_vector)
            probabilities = self._softmax(output_scores)
            target_one_hot = self._one_hot(target_idx)
            expected_one_hot = self._one_hot(context_idx)

            layer_rows.append({
                "index": row_idx,
                "target_word": self.vocabulary[target_idx],
                "context_word": self.vocabulary[context_idx],
                "input_one_hot": self._format_vector(target_one_hot, 0),
                "hidden_vector": self._format_vector(hidden_vector, 4),
                "output_logits": self._format_vector(output_scores, 4),
                "softmax_probabilities": self._format_vector(probabilities, 4),
                "expected_one_hot": self._format_vector(expected_one_hot, 0),
                "target_probability": round(probabilities[context_idx], 6),
            })

        return layer_rows

    def _build_layer_process_cbow(self, pairs: List[Tuple[List[int], int]]) -> List[Dict[str, object]]:
        layer_rows = []

        for row_idx, (context_indices, target_idx) in enumerate(pairs, start=1):
            C = len(context_indices)
            input_vector = [0.0] * len(self.vocabulary)
            for idx in context_indices:
                input_vector[idx] += 1.0 / C

            hidden_vector = [0.0] * self.vector_size
            for idx in context_indices:
                for d in range(self.vector_size):
                    hidden_vector[d] += self.input_weights[idx][d]
            for d in range(self.vector_size):
                hidden_vector[d] /= C

            output_scores = self._compute_output_scores(hidden_vector)
            probabilities = self._softmax(output_scores)
            expected_one_hot = self._one_hot(target_idx)

            context_words = [self.vocabulary[idx] for idx in context_indices]
            target_word = self.vocabulary[target_idx]

            layer_rows.append({
                "index": row_idx,
                "target_word": target_word,
                "context_word": ", ".join(context_words),
                "input_one_hot": self._format_vector(input_vector, 4 if C > 1 else 0),
                "hidden_vector": self._format_vector(hidden_vector, 4),
                "output_logits": self._format_vector(output_scores, 4),
                "softmax_probabilities": self._format_vector(probabilities, 4),
                "expected_one_hot": self._format_vector(expected_one_hot, 0),
                "target_probability": round(probabilities[target_idx], 6),
            })

        return layer_rows

    def _compute_output_scores(self, hidden_vector: List[float]) -> List[float]:
        return [
            self._dot_product(hidden_vector, self.output_weights[word_idx])
            for word_idx in range(len(self.vocabulary))
        ]

    @staticmethod
    def _softmax(scores: List[float]) -> List[float]:
        max_score = max(scores)
        exp_scores = [math.exp(score - max_score) for score in scores]
        total = sum(exp_scores)
        return [score / total for score in exp_scores]

    def _one_hot(self, active_idx: int) -> List[float]:
        return [
            1.0 if idx == active_idx else 0.0
            for idx in range(len(self.vocabulary))
        ]

    @staticmethod
    def _format_vector(vector: List[float], decimals: int) -> str:
        if decimals == 0:
            return "[" + ", ".join(str(int(value)) for value in vector) + "]"
        return "[" + ", ".join(f"{value:.{decimals}f}" for value in vector) + "]"

    @staticmethod
    def _dot_product(vec1: List[float], vec2: List[float]) -> float:
        return sum(a * b for a, b in zip(vec1, vec2))

    def _build_cache_key(self, documents: Dict[str, List[str]]) -> str:
        payload = {
            "params": {
                "window_size": self.window_size,
                "vector_size": self.vector_size,
                "epochs": self.epochs,
                "learning_rate": self.learning_rate,
                "min_count": self.min_count,
                "max_training_pairs": self.max_training_pairs,
                "seed": self.seed,
                "training_type": "full_softmax",
                "architecture": self.architecture,
                "use_library": self.use_library,
            },
            "documents": [
                {
                    "filename": filename,
                    "tokens": documents[filename],
                }
                for filename in sorted(documents)
            ],
        }
        raw_payload = json.dumps(payload, sort_keys=True, ensure_ascii=True)
        return hashlib.sha256(raw_payload.encode("utf-8")).hexdigest()

    def _load_cache(self, cache_key: str) -> bool:
        if not os.path.exists(self.cache_path):
            return False

        try:
            with open(self.cache_path, "r", encoding="utf-8") as cache_file:
                cached = json.load(cache_file)
        except (OSError, json.JSONDecodeError):
            return False

        if cached.get("cache_key") != cache_key:
            return False

        self.vocabulary = cached.get("vocabulary", [])
        self.word_to_idx = {
            word: idx
            for idx, word in enumerate(self.vocabulary)
        }
        self.input_weights = cached.get("input_weights", [])
        self.output_weights = cached.get("output_weights", [])
        self.training_summary = cached.get("training_summary", {})
        self.training_summary["loaded_from_cache"] = True
        self.training_summary["cache_path"] = self.cache_path
        self.trained = True
        return True

    def _save_cache(self, cache_key: str) -> None:
        cache_dir = os.path.dirname(self.cache_path)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)

        payload = {
            "cache_key": cache_key,
            "vocabulary": self.vocabulary,
            "input_weights": self.input_weights,
            "output_weights": self.output_weights,
            "training_summary": self.training_summary,
        }

        with open(self.cache_path, "w", encoding="utf-8") as cache_file:
            json.dump(payload, cache_file, ensure_ascii=False)


class FastTextManual(SkipGramWord2Vec):
    def __init__(self, *args, **kwargs):
        kwargs["architecture"] = "fasttext"
        super().__init__(*args, **kwargs)
        
    def _get_subwords(self, word: str) -> List[str]:
        if len(word) < 3:
            return [word]
        subwords = [word]
        extended = f"<{word}>"
        for i in range(len(extended) - 2):
            subwords.append(extended[i:i+3])
        return list(set(subwords))
        
    def train(self, documents: Dict[str, List[str]]) -> None:
        cache_key = self._build_cache_key(documents)
        if self._load_cache(cache_key):
            print(f"  FastText dimuat dari cache: {self.cache_path}")
            return

        sentences = [tokens for tokens in documents.values() if tokens]
        
        if self.use_library:
            from gensim.models import FastText
            model = FastText(
                sentences,
                vector_size=self.vector_size,
                window=self.window_size,
                epochs=self.epochs,
                min_count=self.min_count,
                seed=self.seed,
                workers=1
            )
            self.vocabulary = model.wv.index_to_key
            self.word_to_idx = {word: idx for idx, word in enumerate(self.vocabulary)}
            self.input_weights = [model.wv[word].tolist() for word in self.vocabulary]
            self.output_weights = []
            
            self.training_summary = {
                "cache_path": self.cache_path,
                "cache_key": cache_key,
                "loaded_from_cache": False,
                "window_size": self.window_size,
                "vector_size": self.vector_size,
                "epochs": self.epochs,
                "learning_rate": self.learning_rate,
                "training_type": "gensim_library",
                "training_architecture": "fasttext",
                "use_library": True,
                "max_training_pairs": self.max_training_pairs,
                "document_count": len(sentences),
                "token_count": sum(len(sentence) for sentence in sentences),
                "vocabulary_size": len(self.vocabulary),
                "pair_count": 0,
                "trained_pair_count": 0,
                "epoch_logs": [{"epoch": epoch + 1, "learning_rate": self.learning_rate, "average_loss": 0.0} for epoch in range(self.epochs)],
            }
            self.trained = True
            self.gensim_model = model
            self._save_cache(cache_key)
            return

        word_counts = Counter(token for sentence in sentences for token in sentence)
        
        valid_words = [token for token, count in word_counts.items() if count >= self.min_count]
        subword_vocab = set()
        for word in valid_words:
            subword_vocab.update(self._get_subwords(word))
            
        self.vocabulary = sorted(list(subword_vocab))
        self.word_to_idx = {sw: idx for idx, sw in enumerate(self.vocabulary)}
        
        pairs = []
        for sentence in sentences:
            for target_pos, target_word in enumerate(sentence):
                if target_word not in valid_words:
                    continue
                start = max(0, target_pos - self.window_size)
                end = min(len(sentence), target_pos + self.window_size + 1)
                for context_pos in range(start, end):
                    if context_pos != target_pos:
                        context_word = sentence[context_pos]
                        if context_word in valid_words:
                            pairs.append((target_word, self.word_to_idx[context_word]))
                            
        self.training_summary = {
            "cache_path": self.cache_path,
            "cache_key": cache_key,
            "loaded_from_cache": False,
            "window_size": self.window_size,
            "vector_size": self.vector_size,
            "epochs": self.epochs,
            "learning_rate": self.learning_rate,
            "training_type": "fasttext_softmax",
            "training_architecture": "fasttext",
            "max_training_pairs": self.max_training_pairs,
            "document_count": len(sentences),
            "token_count": sum(len(sentence) for sentence in sentences),
            "vocabulary_size": len(self.vocabulary),
            "pair_count": len(pairs),
            "trained_pair_count": 0,
            "epoch_logs": [],
        }
        
        if not pairs:
            self.trained = False
            return
            
        training_pairs = pairs[:self.max_training_pairs]
        self.training_summary["trained_pair_count"] = len(training_pairs)
        
        rng = random.Random(self.seed)
        vocab_size = len(self.vocabulary)
        self.input_weights = self._initialize_weights(rng, vocab_size)
        self.output_weights = self._initialize_weights(rng, vocab_size)
        
        for epoch in range(self.epochs):
            rng.shuffle(training_pairs)
            alpha = self.learning_rate * (1 - (epoch / max(1, self.epochs)))
            alpha = max(alpha, self.learning_rate * 0.1)
            epoch_loss = 0.0
            update_count = 0
            
            for target_word, context_idx in training_pairs:
                loss = self._update_softmax_pair_fasttext(target_word, context_idx, alpha)
                epoch_loss += loss
                update_count += 1
                
            average_loss = epoch_loss / max(1, update_count)
            self.training_summary["epoch_logs"].append({
                "epoch": epoch + 1,
                "learning_rate": round(alpha, 6),
                "average_loss": round(average_loss, 6),
            })
            
        self.trained = True
        self.training_summary["trained"] = self.trained
        self._save_cache(cache_key)
        
    def _update_softmax_pair_fasttext(self, target_word: str, context_idx: int, alpha: float) -> float:
        subwords = self._get_subwords(target_word)
        valid_subwords = [sw for sw in subwords if sw in self.word_to_idx]
        if not valid_subwords:
            return 0.0
            
        hidden_vector = [0.0] * self.vector_size
        for sw in valid_subwords:
            sw_idx = self.word_to_idx[sw]
            for d in range(self.vector_size):
                hidden_vector[d] += self.input_weights[sw_idx][d]
        for d in range(self.vector_size):
            hidden_vector[d] /= len(valid_subwords)
            
        output_scores = self._compute_output_scores(hidden_vector)
        probabilities = self._softmax(output_scores)
        loss = -math.log(max(probabilities[context_idx], 1e-12))
        
        output_errors = probabilities[:]
        output_errors[context_idx] -= 1.0
        hidden_error = [0.0] * self.vector_size
        
        for word_idx in range(len(self.vocabulary)):
            for dimension in range(self.vector_size):
                hidden_error[dimension] += output_errors[word_idx] * self.output_weights[word_idx][dimension]
                self.output_weights[word_idx][dimension] -= alpha * output_errors[word_idx] * hidden_vector[dimension]
                
        for sw in valid_subwords:
            sw_idx = self.word_to_idx[sw]
            for dimension in range(self.vector_size):
                self.input_weights[sw_idx][dimension] -= (alpha * hidden_error[dimension]) / len(valid_subwords)
                
        return loss
        
    def get_sentence_vector(self, tokens: List[str], weights: Dict[str, float] = None) -> List[float]:
        vectors = []
        token_weights = []
        for token in tokens:
            word_vec = None
            if hasattr(self, 'gensim_model') and self.gensim_model is not None:
                try:
                    word_vec = self.gensim_model.wv[token].tolist()
                except Exception:
                    pass
            else:
                subwords = self._get_subwords(token)
                valid_subwords = [sw for sw in subwords if sw in self.word_to_idx]
                if valid_subwords:
                    word_vec = [0.0] * self.vector_size
                    for sw in valid_subwords:
                        sw_idx = self.word_to_idx[sw]
                        for d in range(self.vector_size):
                            word_vec[d] += self.input_weights[sw_idx][d]
                    for d in range(self.vector_size):
                        word_vec[d] /= len(valid_subwords)
                    
            if word_vec is not None:
                vectors.append(word_vec)
                if weights is not None:
                    token_weights.append(max(weights.get(token, 0.0), 1e-5))
                else:
                    token_weights.append(1.0)
                
        if not vectors:
            return [0.0] * self.vector_size
            
        total_weight = sum(token_weights)
        return [
            sum(vectors[i][dimension] * token_weights[i] for i in range(len(vectors))) / total_weight
            for dimension in range(self.vector_size)
        ]
