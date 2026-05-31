import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.applications import VGG16  # type: ignore
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

# 1. فحص النموذج ورسم الأداء

def load_data(data_dir):
    images, ages = [], []
    for filename in os.listdir(data_dir):
        if filename.endswith(('.jpg', '.png')):
            age = int(filename.split('_')[0])  # العمر هو الرقم الأول في اسم الملف
            img_path = os.path.join(data_dir, filename)
            img = keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
            img_array = keras.preprocessing.image.img_to_array(img) / 255.0  # تطبيع الصورة
            images.append(img_array)
            ages.append(age)
    return np.array(images), np.array(ages)

# تحميل النموذج المدرب
model_save_path = 'age_estimation_model_VGG16.h5'
model = keras.models.load_model(model_save_path)

test_data_dir = 'T'
X_test, y_test = load_data(test_data_dir)

# إجراء التنبؤات
y_pred = model.predict(X_test)

# حساب مقاييس الأداء
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

# رسم الأداء
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, color='blue', alpha=0.5, label='Predictions')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--', label='Ideal Fit')
plt.xlabel('Actual Age')
plt.ylabel('Predicted Age')
plt.title('Model Performance: Actual vs. Predicted')
plt.legend()
plt.grid(True)

# طباعة القيم
plt.figtext(0.15, 0.01, f"MAE: {mae:.2f} | MSE: {mse:.2f} | RMSE: {rmse:.2f}", fontsize=12, color='black')

# عرض الرسم
plt.show()
