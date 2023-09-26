import os
import pathlib

def save_data(res, filename, dataset_folder_path):
    """Save downloaded data to specified folder."""
    pathlib.Path(dataset_folder_path).mkdir(parents=True, exist_ok=True) 
    save_path = os.path.join(dataset_folder_path, filename)
    with open(save_path, "wb") as file_obj:
        for chunk in res.iter_content(200000):
            file_obj.write(chunk)