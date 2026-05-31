import os
import shutil 
def rename_and_save_images(folder_path, output_folder, reference_year=2015):
    
    for file_name in os.listdir(folder_path):
        
        if not file_name.startswith("nm") or not file_name.endswith((".jpg", ".png")):
            continue
        
        
        parts = file_name.split("_")
        try:
            birth_year = int(parts[2].split("-")[0])  
        except (IndexError, ValueError):
            print(f"Skipping file with unexpected format: {file_name}")
            continue
        
        
        age = reference_year - birth_year
        
        
        new_file_name = f"{age}_0_1_{file_name}"
        old_file_path = os.path.join(folder_path, file_name)
        new_file_path = os.path.join(output_folder, new_file_name)
        
        
        try:
            shutil.copy(old_file_path, new_file_path)
            print(f"Copied and renamed: {file_name} -> {new_file_name}")
        except Exception as e:
            print(f"Error copying {file_name}: {e}")


folder_path = "imge"  # مجلد الصور الأصلي
output_folder = "newimage"  # المجلد الجديد لحفظ الصور بعد إعادة التسمية
rename_and_save_images(folder_path, output_folder)
