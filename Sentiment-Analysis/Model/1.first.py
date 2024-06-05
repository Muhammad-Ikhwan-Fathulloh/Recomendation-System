import pandas as pd
import numpy as np

# Contoh data ulasan
data = {
    'user_id': [1, 2, 3, 1, 2],
    'item_id': [101, 101, 102, 103, 104],
    'rating': [5, 4, 3, 5, 2]
}

df = pd.DataFrame(data)

# Membuat matriks peringkat pengguna-item
rating_matrix = df.pivot(index='user_id', columns='item_id', values='rating').fillna(0)
print(rating_matrix)
