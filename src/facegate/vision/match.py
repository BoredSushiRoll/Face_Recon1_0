import numpy as np

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

def match_face(embedding: np.ndarray, known: list, threshold: float = 0.5):
    """
    known: list of (name, embedding)
    returns (best_name, best_score, is_match)
    """
    best_name, best_score = None, -1.0
    for name, known_emb in known:
        score = cosine_similarity(embedding, known_emb)
        if score > best_score:
            best_name, best_score = name, score
    return best_name, best_score, best_score >= threshold
