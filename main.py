import os
import re
import json
from datetime import datetime
from markupsafe import escape  # pyre-ignore
from typing import Dict, List, Any

from modules.loader import DocumentLoader  # pyre-ignore
from modules.preprocessing import TextPreprocessor  # pyre-ignore
from modules.feature_extraction import FeatureExtractor  # pyre-ignore
from modules.similarity import SimilarityCalculator  # pyre-ignore
from modules.word2vec import SkipGramWord2Vec, FastTextManual  # pyre-ignore


class IRSystem:
    def __init__(self, dataset_path="docs"):
        self.dataset_path = dataset_path
        self.loader = DocumentLoader(dataset_path)
        self.preprocessor = TextPreprocessor()
        self.feature_extractor = FeatureExtractor()
        self.word2vec_sg = SkipGramWord2Vec(window_size=2, architecture="skipgram", cache_path="storage/word2vec_sg_cache.json")
        self.word2vec_cbow = SkipGramWord2Vec(window_size=2, architecture="cbow", cache_path="storage/word2vec_cbow_cache.json")
        self.word2vec_ft = FastTextManual(window_size=2, cache_path="storage/word2vec_ft_cache.json")
        self.similarity_calc = SimilarityCalculator()

        # Data storage
        self.raw_documents = {}
        self.preprocessed_data = {}
        self.doc_tokens = {}
        self.doc_vectors = {}
        self.w2v_sg_doc_vectors = {}
        self.w2v_cbow_doc_vectors = {}
        self.w2v_ft_doc_vectors = {}
        self.doc_names = []
        self.sim_matrix = []
        self.w2v_sg_sim_matrix = []
        self.w2v_cbow_sim_matrix = []
        self.w2v_ft_sim_matrix = []
        self.valid_words = set()
        self.is_initialized = False
        self.current_version = "Default"

    def list_versions(self) -> List[Dict[str, Any]]:
        versions_dir = os.path.join("storage", "versions")
        if not os.path.exists(versions_dir):
            os.makedirs(versions_dir, exist_ok=True)
            return []
        
        versions = []
        for name in os.listdir(versions_dir):
            metadata_path = os.path.join(versions_dir, name, "metadata.json")
            if os.path.exists(metadata_path):
                try:
                    with open(metadata_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                        versions.append(meta)
                except Exception as e:
                    print(f"Error reading version metadata for {name}: {e}")
        
        # Sort by creation time descending (newest first)
        versions.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return versions

    def save_version(self, version_name: str) -> bool:
        versions_dir = os.path.join("storage", "versions", version_name)
        os.makedirs(versions_dir, exist_ok=True)
        
        try:
            # Save metadata.json
            metadata = {
                "version_name": version_name,
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "document_count": len(self.doc_names),
                "vocabulary_size": len(self.word2vec_sg.vocabulary),
                "window_size": self.word2vec_sg.window_size,
                "vector_size": self.word2vec_sg.vector_size,
                "epochs": self.word2vec_sg.epochs,
                "max_training_pairs": self.word2vec_sg.max_training_pairs,
                "use_library": self.word2vec_sg.use_library,
            }
            with open(os.path.join(versions_dir, "metadata.json"), "w", encoding="utf-8") as f:
                json.dump(metadata, f, indent=4, ensure_ascii=False)
                
            # Save corpus.json
            corpus = {
                "raw_documents": self.raw_documents,
                "preprocessed_data": self.preprocessed_data,
                "doc_tokens": self.doc_tokens,
                "doc_names": self.doc_names
            }
            with open(os.path.join(versions_dir, "corpus.json"), "w", encoding="utf-8") as f:
                json.dump(corpus, f, ensure_ascii=False)
                
            # Save tfidf.json
            tfidf = {
                "vocabulary": self.feature_extractor.vocabulary,
                "tf": self.feature_extractor.tf,
                "df": self.feature_extractor.df,
                "idf": self.feature_extractor.idf,
                "tfidf": self.feature_extractor.tfidf,
                "doc_vectors": self.doc_vectors
            }
            with open(os.path.join(versions_dir, "tfidf.json"), "w", encoding="utf-8") as f:
                json.dump(tfidf, f, ensure_ascii=False)
                
            # Save word2vec_sg.json
            w2v_sg = {
                "vocabulary": self.word2vec_sg.vocabulary,
                "input_weights": self.word2vec_sg.input_weights,
                "output_weights": self.word2vec_sg.output_weights,
                "training_summary": self.word2vec_sg.training_summary,
                "w2v_doc_vectors": self.w2v_sg_doc_vectors
            }
            with open(os.path.join(versions_dir, "word2vec_sg.json"), "w", encoding="utf-8") as f:
                json.dump(w2v_sg, f, ensure_ascii=False)

            # Save word2vec_cbow.json
            w2v_cbow = {
                "vocabulary": self.word2vec_cbow.vocabulary,
                "input_weights": self.word2vec_cbow.input_weights,
                "output_weights": self.word2vec_cbow.output_weights,
                "training_summary": self.word2vec_cbow.training_summary,
                "w2v_doc_vectors": self.w2v_cbow_doc_vectors
            }
            with open(os.path.join(versions_dir, "word2vec_cbow.json"), "w", encoding="utf-8") as f:
                json.dump(w2v_cbow, f, ensure_ascii=False)

            # Save word2vec_ft.json
            w2v_ft = {
                "vocabulary": self.word2vec_ft.vocabulary,
                "input_weights": self.word2vec_ft.input_weights,
                "output_weights": self.word2vec_ft.output_weights,
                "training_summary": self.word2vec_ft.training_summary,
                "w2v_doc_vectors": self.w2v_ft_doc_vectors
            }
            with open(os.path.join(versions_dir, "word2vec_ft.json"), "w", encoding="utf-8") as f:
                json.dump(w2v_ft, f, ensure_ascii=False)
                
            # Save dense one_hot_encodings.json mapped to training pairs (matching CSV columns format)
            vocab_size = len(self.word2vec_sg.vocabulary)
            
            # Skip-gram pairs
            sg_xy = self.word2vec_sg.training_summary.get("xy_train", [])
            sg_max = min(len(sg_xy), self.word2vec_sg.max_training_pairs)
            sg_pairs = []
            for idx, row in enumerate(sg_xy[:sg_max]):
                x_idx = row["x_train_index"]
                y_idx = row["y_train_index"]
                x_oh = [0] * vocab_size
                if 0 <= x_idx < vocab_size:
                    x_oh[x_idx] = 1
                y_oh = [0] * vocab_size
                if 0 <= y_idx < vocab_size:
                    y_oh[y_idx] = 1
                sg_pairs.append({
                    "no": row["index"],
                    "x_train_index": x_idx,
                    "x_train_word": row["x_train_word"],
                    "x_train_one_hot": x_oh,
                    "y_train_index": y_idx,
                    "y_train_word": row["y_train_word"],
                    "y_train_one_hot": y_oh
                })
                
            # CBOW pairs
            cbow_xy = self.word2vec_cbow.training_summary.get("xy_train", [])
            cbow_max = min(len(cbow_xy), self.word2vec_cbow.max_training_pairs)
            cbow_pairs = []
            for idx, row in enumerate(cbow_xy[:cbow_max]):
                x_idx = row["x_train_index"] # list
                y_idx = row["y_train_index"]
                x_oh = [0.0] * vocab_size
                if x_idx:
                    for idx_val in x_idx:
                        if 0 <= idx_val < vocab_size:
                            x_oh[idx_val] += 1.0 / len(x_idx)
                y_oh = [0] * vocab_size
                if 0 <= y_idx < vocab_size:
                    y_oh[y_idx] = 1
                cbow_pairs.append({
                    "no": row["index"],
                    "x_train_index": x_idx,
                    "x_train_word": row["x_train_word"],
                    "x_train_one_hot": x_oh,
                    "y_train_index": y_idx,
                    "y_train_word": row["y_train_word"],
                    "y_train_one_hot": y_oh
                })
                
            one_hot_data = {
                "vocabulary_size": vocab_size,
                "vocabulary": self.word2vec_sg.vocabulary,
                "skipgram_pairs": sg_pairs,
                "cbow_pairs": cbow_pairs
            }
            with open(os.path.join(versions_dir, "one_hot_encodings.json"), "w", encoding="utf-8") as f:
                json.dump(one_hot_data, f, ensure_ascii=False)
                
            print(f"[OK] Versi '{version_name}' berhasil disimpan.")
            return True
        except Exception as e:
            print(f"[ERROR] Gagal menyimpan versi '{version_name}': {e}")
            return False

    def load_version(self, version_name: str) -> bool:
        versions_dir = os.path.join("storage", "versions", version_name)
        if not os.path.exists(versions_dir):
            print(f"[ERROR] Versi '{version_name}' tidak ditemukan di storage.")
            return False
            
        try:
            # Load metadata
            with open(os.path.join(versions_dir, "metadata.json"), "r", encoding="utf-8") as f:
                metadata = json.load(f)
            
            # Load corpus
            with open(os.path.join(versions_dir, "corpus.json"), "r", encoding="utf-8") as f:
                corpus = json.load(f)
            self.raw_documents = corpus["raw_documents"]
            self.preprocessed_data = corpus["preprocessed_data"]
            self.doc_tokens = corpus["doc_tokens"]
            self.doc_names = corpus["doc_names"]
            
            # Load tfidf
            with open(os.path.join(versions_dir, "tfidf.json"), "r", encoding="utf-8") as f:
                tfidf = json.load(f)
            self.feature_extractor.vocabulary = tfidf["vocabulary"]
            self.feature_extractor.tf = tfidf["tf"]
            self.feature_extractor.df = tfidf["df"]
            self.feature_extractor.idf = tfidf["idf"]
            self.feature_extractor.tfidf = tfidf["tfidf"]
            self.doc_vectors = tfidf["doc_vectors"]
            
            # Recreate word2vec models
            use_lib = metadata.get("use_library", True)
            self.word2vec_sg = SkipGramWord2Vec(
                window_size=metadata.get("window_size", 2),
                vector_size=metadata.get("vector_size", 10),
                epochs=metadata.get("epochs", 1),
                max_training_pairs=metadata.get("max_training_pairs", 300),
                architecture="skipgram",
                cache_path="storage/word2vec_sg_cache.json",
                use_library=use_lib
            )
            self.word2vec_cbow = SkipGramWord2Vec(
                window_size=metadata.get("window_size", 2),
                vector_size=metadata.get("vector_size", 10),
                epochs=metadata.get("epochs", 1),
                max_training_pairs=metadata.get("max_training_pairs", 300),
                architecture="cbow",
                cache_path="storage/word2vec_cbow_cache.json",
                use_library=use_lib
            )
            self.word2vec_ft = FastTextManual(
                window_size=metadata.get("window_size", 2),
                vector_size=metadata.get("vector_size", 10),
                epochs=metadata.get("epochs", 1),
                max_training_pairs=metadata.get("max_training_pairs", 300),
                cache_path="storage/word2vec_ft_cache.json",
                use_library=use_lib
            )
            
            # Load word2vec_sg weights
            with open(os.path.join(versions_dir, "word2vec_sg.json"), "r", encoding="utf-8") as f:
                w2v_sg = json.load(f)
            self.word2vec_sg.vocabulary = w2v_sg["vocabulary"]
            self.word2vec_sg.word_to_idx = {w: i for i, w in enumerate(self.word2vec_sg.vocabulary)}
            self.word2vec_sg.input_weights = w2v_sg["input_weights"]
            self.word2vec_sg.output_weights = w2v_sg["output_weights"]
            self.word2vec_sg.training_summary = w2v_sg["training_summary"]
            self.word2vec_sg.training_summary["loaded_from_cache"] = True
            self.word2vec_sg.trained = True
            self.w2v_sg_doc_vectors = self.word2vec_sg.get_document_vectors(self.doc_tokens, doc_weights=self.feature_extractor.tfidf)

            # Load word2vec_cbow weights
            with open(os.path.join(versions_dir, "word2vec_cbow.json"), "r", encoding="utf-8") as f:
                w2v_cbow = json.load(f)
            self.word2vec_cbow.vocabulary = w2v_cbow["vocabulary"]
            self.word2vec_cbow.word_to_idx = {w: i for i, w in enumerate(self.word2vec_cbow.vocabulary)}
            self.word2vec_cbow.input_weights = w2v_cbow["input_weights"]
            self.word2vec_cbow.output_weights = w2v_cbow["output_weights"]
            self.word2vec_cbow.training_summary = w2v_cbow["training_summary"]
            self.word2vec_cbow.training_summary["loaded_from_cache"] = True
            self.word2vec_cbow.trained = True
            self.w2v_cbow_doc_vectors = self.word2vec_cbow.get_document_vectors(self.doc_tokens, doc_weights=self.feature_extractor.tfidf)

            # Load word2vec_ft weights
            with open(os.path.join(versions_dir, "word2vec_ft.json"), "r", encoding="utf-8") as f:
                w2v_ft = json.load(f)
            self.word2vec_ft.vocabulary = w2v_ft["vocabulary"]
            self.word2vec_ft.word_to_idx = {w: i for i, w in enumerate(self.word2vec_ft.vocabulary)}
            self.word2vec_ft.input_weights = w2v_ft["input_weights"]
            self.word2vec_ft.output_weights = w2v_ft["output_weights"]
            self.word2vec_ft.training_summary = w2v_ft["training_summary"]
            self.word2vec_ft.training_summary["loaded_from_cache"] = True
            self.word2vec_ft.trained = True
            self.w2v_ft_doc_vectors = self.word2vec_ft.get_document_vectors(self.doc_tokens, doc_weights=self.feature_extractor.tfidf)
            
            # Recompute similarity matrices
            self.sim_matrix = self.similarity_calc.compute_similarity_matrix(
                self.doc_vectors, self.doc_names
            )
            self.w2v_sg_sim_matrix = self.similarity_calc.compute_similarity_matrix(
                self.w2v_sg_doc_vectors, self.doc_names
            )
            self.w2v_cbow_sim_matrix = self.similarity_calc.compute_similarity_matrix(
                self.w2v_cbow_doc_vectors, self.doc_names
            )
            self.w2v_ft_sim_matrix = self.similarity_calc.compute_similarity_matrix(
                self.w2v_ft_doc_vectors, self.doc_names
            )
            
            # Recompute valid words for autocorrect
            self.valid_words = set()
            for d in self.preprocessed_data.values():
                self.valid_words.update(t.lower() for t in d['tokens'] if t.isalpha())
                
            self.current_version = version_name
            self.is_initialized = True
            print(f"[OK] Versi '{version_name}' berhasil dimuat.")
            return True
        except Exception as e:
            print(f"[ERROR] Gagal memuat versi '{version_name}': {e}")
            return False

    def initialize(
        self,
        force_fresh=False,
        architecture="skipgram", # kept for compatibility
        window_size=2,
        vector_size=10,
        epochs=1,
        max_training_pairs=300,
        use_library=True
    ):
        if not force_fresh:
            versions = self.list_versions()
            has_default = any(v['version_name'] == 'Default' for v in versions)
            if has_default:
                print("  Memuat versi 'Default' dari storage...")
                if self.load_version('Default'):
                    return True

        # Re-initialize word2vec extractors with custom training parameters
        self.word2vec_sg = SkipGramWord2Vec(
            window_size=window_size,
            vector_size=vector_size,
            epochs=epochs,
            max_training_pairs=max_training_pairs,
            architecture="skipgram",
            cache_path="storage/word2vec_sg_cache.json",
            use_library=use_library
        )
        self.word2vec_cbow = SkipGramWord2Vec(
            window_size=window_size,
            vector_size=vector_size,
            epochs=epochs,
            max_training_pairs=max_training_pairs,
            architecture="cbow",
            cache_path="storage/word2vec_cbow_cache.json",
            use_library=use_library
        )
        self.word2vec_ft = FastTextManual(
            window_size=window_size,
            vector_size=vector_size,
            epochs=epochs,
            max_training_pairs=max_training_pairs,
            cache_path="storage/word2vec_ft_cache.json",
            use_library=use_library
        )

        print("=" * 50)
        print("TAHAP 1: Memuat Dokumen PDF")
        print("=" * 50)
        self.raw_documents = self.loader.load_all_documents()

        if not self.raw_documents:
            print("[ERROR] Tidak ada dokumen!")
            return False

        self.doc_names = list(self.raw_documents.keys())

        # Preprocessing semua dokumen
        print("=" * 50)
        print("TAHAP 2: Preprocessing Dokumen")
        print("=" * 50)
        doc_tokens = {}
        for i, (filename, text) in enumerate(self.raw_documents.items()):
            print(f"  [{i+1}/{len(self.raw_documents)}] Preprocessing {filename}...")
            result = self.preprocessor.preprocess_detailed(text)
            self.preprocessed_data[filename] = result
            doc_tokens[filename] = result['stemmed_tokens']
        self.doc_tokens = doc_tokens

        # Feature Extraction
        print("=" * 50)
        print("TAHAP 3: Feature Extraction (TF-IDF)")
        print("=" * 50)
        self.feature_extractor.compute_all(doc_tokens)

        # Build document vectors
        for filename in self.doc_names:
            self.doc_vectors[filename] = self.feature_extractor.get_vector(
                self.feature_extractor.tfidf[filename]
            )

        # Word2Vec Feature Extraction
        print("=" * 50)
        print(f"TAHAP 3B: Feature Extraction (Neural Models SKIP-GRAM, CBOW & FastText, window={window_size})")
        print("=" * 50)
        print("  Melatih Word2Vec Skip-gram...")
        self.word2vec_sg.train(doc_tokens)
        self.w2v_sg_doc_vectors = self.word2vec_sg.get_document_vectors(doc_tokens, doc_weights=self.feature_extractor.tfidf)
        
        print("  Melatih Word2Vec CBOW...")
        self.word2vec_cbow.train(doc_tokens)
        self.w2v_cbow_doc_vectors = self.word2vec_cbow.get_document_vectors(doc_tokens, doc_weights=self.feature_extractor.tfidf)

        print("  Melatih FastText...")
        self.word2vec_ft.train(doc_tokens)
        self.w2v_ft_doc_vectors = self.word2vec_ft.get_document_vectors(doc_tokens, doc_weights=self.feature_extractor.tfidf)
        print("  Word2Vec SG, CBOW & FastText selesai dilatih.\n")

        # Hitung similarity matrix antar dokumen
        print("=" * 50)
        print("TAHAP 4: Menghitung Similarity Matrix")
        print("=" * 50)
        self.sim_matrix = self.similarity_calc.compute_similarity_matrix(
            self.doc_vectors, self.doc_names
        )
        self.w2v_sg_sim_matrix = self.similarity_calc.compute_similarity_matrix(
            self.w2v_sg_doc_vectors, self.doc_names
        )
        self.w2v_cbow_sim_matrix = self.similarity_calc.compute_similarity_matrix(
            self.w2v_cbow_doc_vectors, self.doc_names
        )
        self.w2v_ft_sim_matrix = self.similarity_calc.compute_similarity_matrix(
            self.w2v_ft_doc_vectors, self.doc_names
        )
        print("  Similarity matrix selesai dihitung.\n")

        self.is_initialized = True
        print(f"[OK] Sistem siap! {len(self.doc_names)} dokumen dimuat.\n")
        
        # Save as Default
        print("  Menyimpan versi 'Default' ke storage...")
        self.save_version('Default')
        return True

    def search(self, query: str) -> Dict[str, Any]:
        import difflib
        # Kumpulkan daftar kata yang valid dari seluruh dokumen untuk autocorrect
        if not hasattr(self, 'valid_words'):
            self.valid_words = set()
            for d in self.preprocessed_data.values():
                self.valid_words.update(t.lower() for t in d['tokens'] if t.isalpha())

        # Logic Auto-correct
        temp_data = self.preprocessor.preprocess_detailed(query)
        corrected_tokens = []
        is_corrected = False
        
        for token in temp_data['tokens']:
            low_token = token.lower()
            if low_token.isalpha() and low_token not in self.valid_words:
                matches = difflib.get_close_matches(low_token, self.valid_words, n=1, cutoff=0.75)
                if matches:
                    corrected_tokens.append(matches[0])
                    is_corrected = True
                    continue
            corrected_tokens.append(token)
            
        autocorrected_query = " ".join(corrected_tokens) if is_corrected else None
        active_query = autocorrected_query if is_corrected else query

        # Preprocess active query
        query_data = self.preprocessor.preprocess_detailed(active_query)

        # Hitung TF-IDF query
        query_tf, query_tfidf = self.feature_extractor.get_query_tfidf(
            query_data['stemmed_tokens']
        )
        query_vector = self.feature_extractor.get_vector(query_tfidf)
        query_w2v_sg_vector = self.word2vec_sg.get_sentence_vector(
            query_data['stemmed_tokens'], weights=query_tfidf
        )
        query_w2v_cbow_vector = self.word2vec_cbow.get_sentence_vector(
            query_data['stemmed_tokens'], weights=query_tfidf
        )
        query_w2v_ft_vector = self.word2vec_ft.get_sentence_vector(
            query_data['stemmed_tokens'], weights=query_tfidf
        )

        # Hitung similarity query vs setiap dokumen
        results: List[Dict[str, Any]] = []
        for filename in self.doc_names:
            score = self.similarity_calc.cosine_similarity(
                query_vector, self.doc_vectors[filename]
            )
            sg_score = self.similarity_calc.cosine_similarity(
                query_w2v_sg_vector, self.w2v_sg_doc_vectors[filename]
            )
            cbow_score = self.similarity_calc.cosine_similarity(
                query_w2v_cbow_vector, self.w2v_cbow_doc_vectors[filename]
            )
            ft_score = self.similarity_calc.cosine_similarity(
                query_w2v_ft_vector, self.w2v_ft_doc_vectors[filename]
            )
            
            # Buat snippet dinamis berdasarkan kata kunci
            snippet = self._generate_snippet(self.raw_documents[filename], query_data['clean_tokens'] 
            if query_data['clean_tokens'] 
            else query_data['tokens'])
            highlighted = self._highlight_snippet(snippet, query_data['tokens'], query_data['stemmed_tokens'])

            results.append({
                'rank': 0,
                'filename': filename,
                'display_name': filename.replace('_', ' ').replace('.pdf', ''),
                'score': round(score, 6),
                'sg_score': round(sg_score, 6),
                'cbow_score': round(cbow_score, 6),
                'ft_score': round(ft_score, 6),
                'snippet': snippet,
                'highlighted_snippet': highlighted
            })

        # Urutkan berdasarkan skor masing-masing dan hitung rank
        # 1. Rank TF-IDF
        results.sort(key=lambda x: x['score'], reverse=True)
        for i, r in enumerate(results):
            r['rank'] = i + 1

        # 2. Rank Skip-gram
        sg_ranked = sorted(results, key=lambda x: x['sg_score'], reverse=True)
        for i, r in enumerate(sg_ranked):
            r['sg_rank'] = i + 1

        # 3. Rank CBOW
        cbow_ranked = sorted(results, key=lambda x: x['cbow_score'], reverse=True)
        for i, r in enumerate(cbow_ranked):
            r['cbow_rank'] = i + 1

        # 4. Rank FastText
        ft_ranked = sorted(results, key=lambda x: x['ft_score'], reverse=True)
        for i, r in enumerate(ft_ranked):
            r['ft_rank'] = i + 1

        # Hitung jumlah dokumen relevan (score > 0 atau salah satu model > 0)
        relevant_count = sum(1 for r in results if r['score'] > 0 or r['sg_score'] > 0 or r['cbow_score'] > 0 or r['ft_score'] > 0)

        # Data preprocessing untuk template
        doc_prep_summary = self._get_preprocessing_summary()

        # Data feature extraction untuk template (fokus query terms)
        query_terms = list(query_tfidf.keys())
        feat_data = self._get_feature_data(query_terms, query_tf, query_tfidf)

        # Data comparison table
        comparison = self._get_comparison_data(query_terms, query_tfidf, results[:10])  # pyre-ignore

        # Data vektor lengkap (semua dimensi vocabulary)
        full_vectors = self._get_full_vector_data(query_tfidf, results[:10])  # pyre-ignore

        # Nama dokumen pendek untuk similarity matrix
        short_names = [fn.replace('_', ' ').replace('.pdf', '') for fn in self.doc_names]

        method_comparison = self._get_method_comparison(results)

        return {
            'results': results,
            'sg_results': sg_ranked,
            'cbow_results': cbow_ranked,
            'ft_results': ft_ranked,
            'relevant_count': relevant_count,
            'autocorrected_query': autocorrected_query,
            'original_query': query,
            'query_preprocessing': {
                'original': query,
                'tokens': query_data['tokens'],
                'clean_tokens': query_data['clean_tokens'],
                'stemmed_tokens': query_data['stemmed_tokens']
            },
            'query_tf': query_tf,
            'query_tfidf': query_tfidf,
            'doc_preprocessing': doc_prep_summary,
            'feature_data': feat_data,
            'comparison': comparison,
            'full_vectors': full_vectors,
            'method_comparison': method_comparison,
            'sim_matrix': self.sim_matrix,
            'w2v_sg_sim_matrix': self.w2v_sg_sim_matrix,
            'w2v_cbow_sim_matrix': self.w2v_cbow_sim_matrix,
            'w2v_ft_sim_matrix': self.w2v_ft_sim_matrix,
            'sim_doc_names': short_names,
        }

    def _generate_snippet(self, text: str, query_tokens: List[str], window: int = 150) -> str:
        text_lower = text.lower()
        best_idx = -1
        
        # Cari term terpanjang yang cocok untuk menjadi fokus snippet
        for token in sorted(query_tokens, key=len, reverse=True):
            if len(token) > 2:
                idx = text_lower.find(token.lower())
                if idx != -1:
                    best_idx = idx
                    break
                    
        # Jika tidak ada yang cocok secara persis (mungkin kena lewat stemming aja), ambil awal teks
        if best_idx == -1:
            return text[:300].replace('\n', ' ')  # pyre-ignore
            
        start = max(0, best_idx - window)
        end = min(len(text), best_idx + len(token) + window)
        
        snippet = text[start:end].replace('\n', ' ')  # pyre-ignore
        
        if start > 0:
            snippet = "..." + snippet.lstrip()
        if end < len(text):
            snippet = snippet.rstrip() + "..."
            
        return snippet

    def _highlight_snippet(self, snippet, query_tokens, query_stemmed_tokens):
        import math
        safe_snippet = str(escape(snippet))
        
        # Lowercase set of raw query tokens and stemmed query tokens for quick lookup
        raw_query_set = {t.lower() for t in query_tokens if len(t) > 1}
        stemmed_query_set = {t.lower() for t in query_stemmed_tokens if len(t) > 1}
        
        # 1. Temukan seluruh kata dalam snippet
        word_pattern = re.compile(r'\b([a-zA-Z0-9_\-]+)\b')
        all_words = word_pattern.findall(safe_snippet)
        
        # Saring kandidat kata unik yang bukan exact match
        candidate_words = []
        for word in set(all_words):
            word_lower = word.lower()
            if len(word_lower) <= 1:
                continue
            if word_lower in raw_query_set:
                continue
            word_stemmed = self.preprocessor.stem_word(word_lower)
            if word_stemmed in stemmed_query_set:
                continue
            candidate_words.append((word, word_lower, word_stemmed))
            
        # 2. Hitung kemiripan kosinus untuk seluruh kandidat kata
        semantic_matches = []
        for word, word_lower, word_stemmed in candidate_words:
            word_vec = self.word2vec_ft.get_sentence_vector([word_stemmed])
            mag = math.sqrt(sum(x**2 for x in word_vec))
            if mag < 1e-6:
                continue
                
            max_sim = 0.0
            best_query_term = None
            for q_term in stemmed_query_set:
                q_vec = self.word2vec_ft.get_sentence_vector([q_term])
                q_mag = math.sqrt(sum(x**2 for x in q_vec))
                if q_mag < 1e-6:
                    continue
                dot = sum(a*b for a, b in zip(word_vec, q_vec))
                sim = dot / (mag * q_mag)
                if sim > max_sim:
                    max_sim = sim
                    best_query_term = q_term
            
            # Hanya masukkan jika kemiripan cukup signifikan (> 0.72)
            if max_sim > 0.72 and best_query_term is not None:
                semantic_matches.append({
                    'word': word,
                    'word_lower': word_lower,
                    'sim': max_sim,
                    'matched_term': best_query_term
                })
                
        # 3. Urutkan berdasarkan kemiripan tertinggi dan batasi maksimal 3 kata semantik teratas
        semantic_matches.sort(key=lambda x: x['sim'], reverse=True)
        top_semantic = semantic_matches[:3]
        top_semantic_words = {x['word_lower']: x for x in top_semantic}
        
        # 4. Ganti kata kunci di dalam snippet dengan kelas highlight yang sesuai
        def replace_match(match):
            word = match.group(1)
            word_lower = word.lower()
            if len(word_lower) <= 1:
                return word
                
            # Exact match (raw atau stemmed)
            if word_lower in raw_query_set:
                return f'<mark class="exact-match">{word}</mark>'
            word_stemmed = self.preprocessor.stem_word(word_lower)
            if word_stemmed in stemmed_query_set:
                return f'<mark class="exact-match">{word}</mark>'
                
            # Semantic match (terbatas hanya untuk top 3)
            if word_lower in top_semantic_words:
                match_data = top_semantic_words[word_lower]
                tooltip = f"Semantic match to &#39;{match_data['matched_term']}&#39; (Similarity: {match_data['sim']:.2f})"
                return f'<mark class="semantic-match" title="{tooltip}">{word}</mark>'
                
            return word

        return word_pattern.sub(replace_match, safe_snippet)

    def _get_preprocessing_summary(self):
        summaries = []
        for filename in self.doc_names:
            d = self.preprocessed_data[filename]
            summaries.append({
                'filename': filename,
                'display_name': filename.replace('_', ' ').replace('.pdf', ''),
                'original_snippet': d['original'][:200],
                'token_count': len(d['tokens']),
                'clean_token_count': len(d['clean_tokens']),
                'stemmed_token_count': len(d['stemmed_tokens']),
                'tokens_sample': d['tokens'][:15],
                'clean_tokens_sample': d['clean_tokens'][:15],
                'stemmed_tokens_sample': d['stemmed_tokens'][:15],
            })
        return summaries

    def _print_word2vec_training_summary(self):
        # Print Skip-gram summary
        summary_sg = self.word2vec_sg.training_summary
        if summary_sg:
            print("  --- Skip-gram Training Summary ---")
            print(f"  Window size: {summary_sg['window_size']}")
            print(f"  Vector size: {summary_sg['vector_size']}")
            print(f"  Epochs: {summary_sg['epochs']}")
            print(f"  Total token training: {summary_sg['token_count']}")
            print(f"  Jumlah pasangan target-context: {summary_sg['pair_count']}")
            
        # Print CBOW summary
        summary_cbow = self.word2vec_cbow.training_summary
        if summary_cbow:
            print("  --- CBOW Training Summary ---")
            print(f"  Window size: {summary_cbow['window_size']}")
            print(f"  Vector size: {summary_cbow['vector_size']}")
            print(f"  Epochs: {summary_cbow['epochs']}")
            print(f"  Total token training: {summary_cbow['token_count']}")
            print(f"  Jumlah pasangan target-context: {summary_cbow['pair_count']}")

    def _get_feature_data(self, query_terms, query_tf, query_tfidf):
        # Tabel TF-IDF query
        query_table = []
        for term in query_terms:
            query_table.append({
                'term': term,
                'tf': query_tf.get(term, 0),
                'idf': round(self.feature_extractor.idf.get(term, 0), 4),
                'tfidf': round(query_tfidf.get(term, 0), 4)
            })

        # Tabel TF dokumen untuk query terms
        doc_tf_rows: List[Dict[str, Any]] = []
        for filename in self.doc_names:
            row = {
                'filename': filename,
                'display_name': filename.replace('_', ' ').replace('.pdf', ''),
                'values': {}
            }
            for term in query_terms:
                row['values'][term] = self.feature_extractor.tf[filename].get(term, 0)  # pyre-ignore
            doc_tf_rows.append(row)

        # Tabel TF-IDF dokumen untuk query terms
        doc_tfidf_rows: List[Dict[str, Any]] = []
        for filename in self.doc_names:
            row = {
                'filename': filename,
                'display_name': filename.replace('_', ' ').replace('.pdf', ''),
                'values': {}
            }
            for term in query_terms:
                val = self.feature_extractor.tfidf[filename].get(term, 0)
                row['values'][term] = round(val, 4)  # pyre-ignore
            doc_tfidf_rows.append(row)

        return {
            'vocabulary_size': len(self.feature_extractor.vocabulary),
            'query_terms': query_terms,
            'query_table': query_table,
            'doc_tf_rows': doc_tf_rows,
            'doc_tfidf_rows': doc_tfidf_rows,
        }

    def _get_comparison_data(self, query_terms, query_tfidf, top_results):
        """Mendapatkan data tabel perbandingan vektor query vs dokumen."""
        rows: List[Dict[str, Any]] = []
        for term in query_terms:
            row = {
                'term': term,
                'query_value': round(query_tfidf.get(term, 0), 4),
                'doc_values': {}
            }
            for r in top_results:
                fn = r['filename']
                val = self.feature_extractor.tfidf[fn].get(term, 0)
                row['doc_values'][fn] = round(val, 4)  # pyre-ignore
            rows.append(row)

        return {
            'terms': query_terms,
            'top_docs': [{'filename': r['filename'],
                          'display_name': r['display_name']} for r in top_results],
            'rows': rows
        }

    def _get_full_vector_data(self, query_tfidf, top_results):
        """Mendapatkan vektor lengkap (semua dimensi vocabulary) untuk query dan dokumen."""
        vocabulary = self.feature_extractor.vocabulary
        query_vector = self.feature_extractor.get_vector(query_tfidf)

        # Buat rows: setiap baris = 1 term dari vocabulary
        rows = []
        for idx, term in enumerate(vocabulary):
            q_val = round(query_vector[idx], 4)
            doc_vals = {}
            for r in top_results:
                fn = r['filename']
                val = self.feature_extractor.tfidf[fn].get(term, 0)
                doc_vals[fn] = round(val, 4)
            rows.append({
                'index': idx + 1,
                'term': term,
                'query_value': q_val,
                'doc_values': doc_vals,
                'is_nonzero': q_val > 0 or any(v > 0 for v in doc_vals.values())
            })

        return {
            'vocabulary_size': len(vocabulary),
            'top_docs': [{'filename': r['filename'],
                          'display_name': r['display_name']} for r in top_results],
            'rows': rows
        }

    def _get_method_comparison(self, results):
        """Membandingkan ranking TF-IDF, Word2Vec Skip-gram, CBOW dan FastText."""
        rows = []
        for r in results:
            rows.append({
                'filename': r['filename'],
                'display_name': r['display_name'],
                'tfidf_rank': r['rank'],
                'tfidf_score': r['score'],
                'sg_rank': r['sg_rank'],
                'sg_score': r['sg_score'],
                'cbow_rank': r['cbow_rank'],
                'cbow_score': r['cbow_score'],
                'ft_rank': r['ft_rank'],
                'ft_score': r['ft_score'],
            })

        return {
            'word2vec_window': self.word2vec_sg.window_size,
            'word2vec_vector_size': self.word2vec_sg.vector_size,
            'word2vec_sg_training': self.word2vec_sg.training_summary,
            'word2vec_cbow_training': self.word2vec_cbow.training_summary,
            'word2vec_ft_training': self.word2vec_ft.training_summary,
            'rows': rows,
        }

    def _calculate_metrics(self, retrieved_list, ground_truth):
        if not ground_truth:
            return {
                'top1_accuracy': 0.0,
                'precision': 0.0,
                'recall': 0.0,
                'f1_score': 0.0,
                'map': 0.0
            }
        
        gt_set = set(ground_truth)
        
        # Top-1 Accuracy
        top1_accuracy = 1.0 if retrieved_list and retrieved_list[0] in gt_set else 0.0
        
        # Precision@5
        top_5 = retrieved_list[:5]
        relevant_retrieved = [doc for doc in top_5 if doc in gt_set]
        precision = len(relevant_retrieved) / 5.0
        
        # Recall@5
        recall = len(relevant_retrieved) / len(gt_set)
        
        # F1-Score
        if precision + recall > 0:
            f1_score = (2.0 * precision * recall) / (precision + recall)
        else:
            f1_score = 0.0
            
        # MAP
        ap = 0.0
        relevant_count = 0
        for k, doc in enumerate(retrieved_list):
            if doc in gt_set:
                relevant_count += 1
                precision_at_k = relevant_count / (k + 1)
                ap += precision_at_k
        ap = ap / len(gt_set) if len(gt_set) > 0 else 0.0
        
        return {
            'top1_accuracy': top1_accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'map': ap
        }

    def evaluate_global_benchmark(self) -> Dict[str, Any]:
        gt_file = os.path.join("storage", "ground_truth.json")
        if not os.path.exists(gt_file):
            return {}
            
        try:
            with open(gt_file, "r") as f:
                gt_data = json.load(f)
        except Exception:
            return {}
            
        if not gt_data:
            return {}
            
        models = ['tfidf', 'sg', 'cbow', 'ft']
        sums = {m: {'top1_accuracy': 0.0, 'precision': 0.0, 'recall': 0.0, 'f1_score': 0.0, 'map': 0.0} for m in models}
        
        query_count = 0
        for query, gt_docs in gt_data.items():
            if not gt_docs:
                continue
            query_count += 1
            
            # Run search silently
            res = self.search(query)
            
            # Retrieve documents
            tfidf_retrieved = [r['filename'] for r in res['results']]
            sg_retrieved = [r['filename'] for r in res['sg_results']]
            cbow_retrieved = [r['filename'] for r in res['cbow_results']]
            ft_retrieved = [r['filename'] for r in res['ft_results']]
            
            # Calculate metrics
            metrics = {
                'tfidf': self._calculate_metrics(tfidf_retrieved, gt_docs),
                'sg': self._calculate_metrics(sg_retrieved, gt_docs),
                'cbow': self._calculate_metrics(cbow_retrieved, gt_docs),
                'ft': self._calculate_metrics(ft_retrieved, gt_docs)
            }
            
            # Sum up
            for m in models:
                for metric in sums[m]:
                    sums[m][metric] += metrics[m][metric]
                    
        if query_count == 0:
            return {}
            
        averages = {}
        for m in models:
            averages[m] = {}
            for metric in sums[m]:
                averages[m][metric] = round(sums[m][metric] / query_count, 4)
                
        return averages

    def load_query_history(self) -> List[str]:
        history_file = os.path.join("storage", "query_history.json")
        if os.path.exists(history_file):
            try:
                with open(history_file, "r") as f:
                    return json.load(f)
            except Exception:
                pass
        return []

    def add_to_query_history(self, query: str):
        query = query.strip()
        if not query:
            return
        history = self.load_query_history()
        if query in history:
            history.remove(query)
        history.insert(0, query)
        history = history[:10]  # Limit to 10 items
        
        os.makedirs("storage", exist_ok=True)
        history_file = os.path.join("storage", "query_history.json")
        try:
            with open(history_file, "w") as f:
                json.dump(history, f, indent=4)
        except Exception as e:
            print(f"Error saving query history: {e}")


# Testing langsung jika file ini dijalankan
if __name__ == "__main__":
    dataset_path = os.path.join(os.path.dirname(__file__), "docs")
    ir = IRSystem(dataset_path)
    ir.initialize()

    while True:
        query = input("\nMasukkan query (ketik 'exit' untuk keluar): ").strip()
        if query.lower() == 'exit':
            break
        if not query:
            continue

        result = ir.search(query)
        print(f"\n--- Top 5 Dokumen Relevan untuk '{query}' ---")
        print("TF-IDF ranking:")
        for r in result['results'][:5]:
            print(f"  {r['rank']}. {r['display_name']} (skor: {r['score']})")

        print("\nWord2Vec Skip-gram ranking (window=2):")
        for r in result['w2v_results'][:5]:
            print(f"  {r['w2v_rank']}. {r['display_name']} (skor: {r['w2v_score']})")
