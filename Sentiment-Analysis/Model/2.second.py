from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

# Contoh data ulasan teks
reviews = ["Ulasan 1", "Ulasan 2", "Ulasan 3"]

# Membuat vektor ulasan
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(reviews)

# LDA
lda = LatentDirichletAllocation(n_components=2, random_state=42)
lda.fit(X)

# Menampilkan topik
for index, topic in enumerate(lda.components_):
    print(f'Topik {index}:')
    print([vectorizer.get_feature_names_out()[i] for i in topic.argsort()[-10:]])
