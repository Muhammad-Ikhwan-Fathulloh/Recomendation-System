# Sentiment Analysis

## Deskripsi Proyek
Proyek ini bertujuan untuk meningkatkan akurasi sistem rekomendasi dengan memanfaatkan berbagai teknik dan model, seperti matriks peringkat pengguna-item, analisis sentimen dengan CNN, dan pemrosesan topik dengan LDA.

## Fitur Utama
- **Matriks Peringkat Pengguna-Item**: Digunakan untuk merepresentasikan preferensi pengguna terhadap item.
- **Analisis Sentimen dengan CNN**: Model CNN digunakan untuk menganalisis sentimen dari ulasan pengguna.
- **Pemrosesan Topik dengan LDA**: Model LDA digunakan untuk mengidentifikasi topik utama dari ulasan pengguna.
- **Penanganan Data Imbalance dengan SMOTE**: Teknik SMOTE digunakan untuk menangani ketidakseimbangan dalam data sentimen.
- **Collaborative Filtering**: Menggunakan cosine similarity untuk membandingkan preferensi antara pengguna.

## Contoh Kode
```python
# Matriks Peringkat Pengguna-Item
import numpy as np

user_item_matrix = np.array([[5, 0, 3],
                             [4, 0, 0],
                             [0, 2, 0],
                             [0, 0, 5]])

print("User-Item Matrix:")
print(user_item_matrix)

# Analisis Sentimen dengan CNN
from keras.models import Sequential
from keras.layers import Dense, Conv1D, GlobalMaxPooling1D

model = Sequential()
model.add(Conv1D(128, 5, activation='relu', input_shape=(100, 1)))
model.add(GlobalMaxPooling1D())
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Pemrosesan Topik dengan LDA
from gensim import corpora, models

documents = ["Sangat menyukai tempat ini. Pelayanan sangat ramah.",
             "Makanan enak tetapi pelayanan buruk.",
             "Tempat yang nyaman untuk bersantai.",
             "Tidak puas dengan pengalaman ini."]

dictionary = corpora.Dictionary([doc.lower().split() for doc in documents])
corpus = [dictionary.doc2bow(doc.lower().split()) for doc in documents]
lda_model = models.LdaModel(corpus, num_topics=2, id2word=dictionary)

# Resampling dengan SMOTE
from imblearn.over_sampling import SMOTE

X_resampled, y_resampled = SMOTE().fit_resample(X, y)

# Collaborative Filtering
from sklearn.metrics.pairwise import cosine_similarity

user_similarity = cosine_similarity(user1_features, user2_features)
