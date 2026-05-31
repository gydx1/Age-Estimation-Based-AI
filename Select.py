import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tkinter import Tk
from tkinter.filedialog import askopenfilename
from tensorflow.keras.models import load_model # type: ignore
# تحميل النموذج
model = load_model('age_estimation_model_VGG16.h5')

# دالة لتحميل الصورة وتحضيرها للنموذج
def preprocess_image(image_path):
    img = keras.preprocessing.image.load_img(image_path, target_size=(224,224))
    img_array = keras.preprocessing.image.img_to_array(img) / 255.0  # تطبيع الصورة
    img_array = np.expand_dims(img_array, axis=0)  # إضافة بعد لتتناسب مع إدخال النموذج
    return img_array

# دالة للتنبؤ بالعمر من صورة معينة
def predict_age(image_path):
    img_array = preprocess_image(image_path)
    predicted_age = model.predict(img_array)
    return predicted_age[0][0] # استخراج العمر المقدر

# فتح نافذة اختيار الصورة
Tk().withdraw()  # إخفاء النافذة الرئيسية لتطبيق tkinter
file_path = askopenfilename(title="اختر صورة", filetypes=[("Image Files", "*.jpg;*.png")])

# التنبؤ بالعمر إذا تم اختيار صورة
if file_path:
    predicted_age = predict_age(file_path)
    print(f"Predicted Age: {predicted_age}")
else:
    print("لم يتم اختيار صورة.")