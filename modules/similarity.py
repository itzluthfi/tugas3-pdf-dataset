import math


class SimilarityCalculator:
    def cosine_similarity(self, vec1, vec2):
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a ** 2 for a in vec1))
        magnitude2 = math.sqrt(sum(b ** 2 for b in vec2))

        if magnitude1 == 0 or magnitude2 == 0:
            return 0.0
        return dot_product / (magnitude1 * magnitude2)

    def euclidean_distance(self, vec1, vec2):
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(vec1, vec2)))

    def compute_similarity_matrix(self, doc_vectors, doc_names):
        n = len(doc_names)
        matrix = []
        for i in range(n):
            row = []
            for j in range(n):
                score = self.cosine_similarity(doc_vectors[doc_names[i]], doc_vectors[doc_names[j]])
                row.append(round(score, 4))  # pyre-ignore
            matrix.append(row)
        return matrix
