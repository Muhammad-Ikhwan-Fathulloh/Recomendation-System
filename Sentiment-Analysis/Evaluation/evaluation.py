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
