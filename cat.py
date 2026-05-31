import os
import shutil
from collections import defaultdict

source_folder = "train"  
target_folder = "remove"  


age_to_files = defaultdict(list)


image_files = [file for file in os.listdir(source_folder) if file.endswith(('.jpg', '.png', '.jpeg'))]


for image in image_files:
    age = image.split('_')[0] 
    age_to_files[age].append(image)


for age, files in age_to_files.items():
    if len(files) > 50:
        
        age_folder = os.path.join(target_folder, f"age_{age}")
        os.makedirs(age_folder, exist_ok=True)

        
        extra_files = files[50:]

        
        for image in extra_files:
            shutil.move(os.path.join(source_folder, image), os.path.join(age_folder, image))

print("تم نقل الصور الزائدة التي تجاوزت 50 لكل عمر.")
