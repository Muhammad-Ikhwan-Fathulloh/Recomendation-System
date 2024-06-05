from sklearn.metrics.pairwise import cosine_similarity

# Matriks peringkat pengguna-item
user_item_matrix = np.array([[5, 0, 3], [4, 2, 1], [1, 5, 0]])

# Kesamaan pengguna menggunakan cosine similarity
user_similarity = cosine_similarity(user_item_matrix)
print(user_similarity)