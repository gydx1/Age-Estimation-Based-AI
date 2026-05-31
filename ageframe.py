import cv2
import numpy as np
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model  # type: ignore
import tkinter as tk
from tkinter import Label, Canvas
import os
# تحميل النموذج المدرب
#model = load_model('age_estimation_model_vgg16.h5')
model_path = os.path.join(os.path.dirname(__file__), "age_estimation_model_VGG16.h5")
model = load_model(model_path)
# إعداد الكاميرا
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error opening camera")
    exit()

# إنشاء نافذة tkinter بحجم أصغر
root = tk.Tk()
root.title("Age Estimation")
root.geometry("400x500")  # زيادة ارتفاع النافذة قليلاً لتناسب الدائرة والنص

# عرض صورة الكاميرا في tkinter بحجم أصغر
camera_label = Label(root, width=300, height=200)  # تقليل حجم الإطار
camera_label.pack()

# إنشاء دائرة لعرض العمر
canvas = Canvas(root, width=150, height=150, bg="white", highlightthickness=0)
canvas.pack(pady=20)

# رسم دائرة خضراء
canvas.create_oval(10, 10, 140, 140, fill="green")

# إضافة نص داخل الدائرة لعرض العمر
age_text = canvas.create_text(75, 75, text="N/A", fill="white", font=("Arial", 16, "bold"))

# متغيرات لتخزين الإطار السابق وحالة الكشف عن شخص
previous_frame = None
person_detected = False

# دالة للكشف عن وجود شخص في الإطار
#face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
face_cascade = cv2.CascadeClassifier(os.path.join(os.path.dirname(__file__), "haarcascade_frontalface_default.xml"))


def detect_person(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return len(faces) > 0

# دالة لتحديث الإطار وعرضه
def update_frame():
    global previous_frame, person_detected
    ret, frame = cap.read()
    if ret:
        frame = cv2.resize(frame, (300, 200))  # تغيير حجم الصورة لتناسب الإطار
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        camera_label.imgtk = imgtk
        camera_label.configure(image=imgtk)

        # التحقق مما إذا كان هناك شخص في الإطار
        person_present = detect_person(frame)

        if person_present and not person_detected:
            predict_age(frame)  # إجراء التنبؤ فقط عند ظهور شخص جديد
        elif not person_present:
            canvas.itemconfig(age_text, text="N/A")  # إعادة ضبط النص عند اختفاء الشخص

        person_detected = person_present
    
    root.after(500, update_frame)  # تحديث كل 500 مللي ثانية

# دالة للتنبؤ بالعمر
def predict_age(frame):
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    img = img.resize((224, 224))
    
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predicted_age = model.predict(img_array)
    age_value = f"{predicted_age[0][0]:.2f}"
    
    # تحديث النص في منتصف الدائرة بالقيمة المتوقعة
    canvas.itemconfig(age_text, text=age_value)

# بدء تحديث الإطار
update_frame()

# تشغيل واجهة tkinter
root.mainloop()

# تحرير الكاميرا بعد إغلاق النافذة
cap.release()
cv2.destroyAllWindows()
