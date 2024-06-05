# Evaluasi Performa Model: MSE, RMSE, dan MAE

## Pendahuluan
Dalam evaluasi performa model, terutama pada sistem rekomendasi, tiga metrik yang umum digunakan adalah Mean Squared Error (MSE), Root Mean Squared Error (RMSE), dan Mean Absolute Error (MAE). Masing-masing metrik ini memiliki kelebihan dan kekurangan tersendiri, yang akan dijelaskan secara rinci di bawah ini.

## 1. Mean Squared Error (MSE)

**Rumus:**
\[ \text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2 \]

**Penjelasan:**

- MSE mengukur rata-rata kesalahan kuadrat antara nilai prediksi (\(\hat{y}_i\)) dan nilai sebenarnya (\(y_i\)).
- MSE memperbesar kesalahan besar lebih dari kesalahan kecil karena kesalahan dikuadratkan, sehingga lebih sensitif terhadap outlier.
- Biasanya digunakan dalam tahap pelatihan model untuk mendapatkan model yang paling akurat.

## 2. Root Mean Squared Error (RMSE)

**Rumus:**
\[ \text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2} \]

**Penjelasan:**

- RMSE adalah akar kuadrat dari MSE, sehingga dalam satuan yang sama dengan variabel yang diprediksi.
- Sama seperti MSE, RMSE memperbesar pengaruh outlier karena kuadrat kesalahan, namun lebih mudah diinterpretasikan karena dalam satuan yang sama dengan nilai asli.
- RMSE sering digunakan untuk mengevaluasi performa akhir model karena memberikan gambaran yang jelas tentang kesalahan prediksi.

## 3. Mean Absolute Error (MAE)

**Rumus:**
\[ \text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i| \]

**Penjelasan:**

- MAE mengukur rata-rata kesalahan absolut antara nilai prediksi (\(\hat{y}_i\)) dan nilai sebenarnya (\(y_i\)).
- MAE lebih robust terhadap outlier dibandingkan MSE dan RMSE karena tidak memperbesar kesalahan besar secara eksponensial.
- Lebih mudah diinterpretasikan karena dalam satuan yang sama dengan variabel yang diprediksi.
- Sering digunakan sebagai metrik tambahan untuk memberikan perspektif berbeda tentang performa model.

## Pemilihan Metrik

- **MSE:** Digunakan dalam proses pelatihan model karena memperbesar kesalahan besar dan membantu model untuk lebih hati-hati terhadap outlier.
- **RMSE:** Cocok untuk evaluasi akhir karena memberikan nilai kesalahan dalam satuan yang sama dengan data asli dan lebih sensitif terhadap kesalahan besar, memberikan gambaran yang jelas tentang performa model.
- **MAE:** Digunakan untuk memberikan perspektif yang lebih robust terhadap outlier dan lebih mudah diinterpretasikan.

## Contoh Implementasi

Berikut adalah contoh implementasi dalam Python untuk menghitung ketiga metrik evaluasi:

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

# Contoh nilai asli dan prediksi
y_true = np.array([3.5, 4.0, 3.0, 5.0, 4.5])
y_pred = np.array([3.7, 3.9, 2.8, 5.2, 4.4])

# Menghitung MSE
mse = mean_squared_error(y_true, y_pred)
print(f'MSE: {mse}')

# Menghitung RMSE
rmse = np.sqrt(mse)
print(f'RMSE: {rmse}')

# Menghitung MAE
mae = mean_absolute_error(y_true, y_pred)
print(f'MAE: {mae}')