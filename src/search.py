from sklearn.metrics.pairwise import cosine_similarity


def search(preprocessed_query, tfidf_matrix, vectorizer, top_n=5):
    query_str = " ".join(preprocessed_query)
    query_vec = vectorizer.transform([query_str])

    cosine_similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()

    top_n_idx = cosine_similarities.argsort()[-top_n:][::-1]

    return top_n_idx, cosine_similarities
