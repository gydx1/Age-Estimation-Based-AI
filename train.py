import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.applications import VGG16 # type: ignore
from keras import layers

# 1. جمع بيانات التدريب
train_data_dir = 'train'  # استبدل هذا بمسار مجموعة بيانات التدريب
test_data_dir = 'test'    # استبدل هذا بمسار مجموعة بيانات الاختبار

def load_data(data_dir):
    images = []
    ages = []
    
    # تحميل الصور والأعمار
    for filename in os.listdir(data_dir):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            # استخراج العمر من اسم الملف (الرقم الأول)
            age = int(filename.split('_')[0])  # العمر هو الرقم الأول في اسم الملف
            img_path = os.path.join(data_dir, filename)
            img = keras.preprocessing.image.load_img(img_path, target_size=(224, 224))
            img_array = keras.preprocessing.image.img_to_array(img) / 255.0  # تطبيع الصورة
            images.append(img_array)
            ages.append(age)

    return np.array(images), np.array(ages)

# تحميل بيانات التدريب والاختبار
X_train, y_train = load_data(train_data_dir)
X_test, y_test = load_data(test_data_dir)

# 2. بناء النموذج باستخدام VGG16 الأصلية
def build_model():
    # تحميل VGG16 بالكامل مع الأوزان الأصلية
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    
    # تجميد الطبقات الأساسية (أي الطبقات التي تم تدريبها مسبقًا)
    for layer in base_model.layers:
        layer.trainable = False

    model = keras.Sequential()
    model.add(base_model)
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.GlobalAveragePooling2D())  # بديل لـ Flatten
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.4))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Dropout(0.3))

    model.add(layers.Dense(1, activation='relu'))  # لتكون القيم غير سالبة

    # إعداد النموذج الأولي
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  loss='mean_squared_error', metrics=['mae'])
    
    # تدريب النموذج على البيانات الأساسية
    model.fit(X_train, y_train, epochs=30, validation_split=0.2, batch_size=32)

    # الآن فتح الطبقات الأخيرة في VGG16 للتدريب عليها
    for layer in base_model.layers[-4:]:  # فتح آخر 4 طبقات، يمكنك تعديل هذا الرقم حسب الحاجة
        layer.trainable = True

    # إعادة تجميع النموذج مع التعلم المعدل
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-5),  # استخدام معدل تعلم منخفض
                  loss='mean_squared_error', metrics=['mae'])

    return model



# بناء النموذج
model = build_model()

# 3. تدريب النموذج باستخدام بيانات التدريب
history = model.fit(X_train, y_train, epochs=30, validation_split=0.2, batch_size=32)

# 4. تقييم النموذج باستخدام بيانات الاختبار
test_loss, test_mae = model.evaluate(X_test, y_test)
print(f'Test MAE: {test_mae}')

# 5. حفظ النموذج بعد انتهاء التدريب
model_save_path = 'age_estimation_model_vgg16.h5'  # اسم ملف الحفظ
model.save(model_save_path)
print(f'Model saved to {model_save_path}')
