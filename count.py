import os
from collections import Counter

dataset_path = "train"

ages = []
for filename in os.listdir(dataset_path):
    if filename.endswith(('.jpg', '.jpeg', '.png')):  
        age = filename.split('_')[0]
        ages.append(age)
        
age_counts = Counter(ages)

print("عدد الصور لكل فئة عمرية:")
for age, count in sorted(age_counts.items(), key=lambda x: int(x[0])):
    print(f"العمر {age}: {count} صورة")
