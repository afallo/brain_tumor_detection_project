import pandas as pd
import os


def load_path_label_df(data_dir) :
    # Collect image paths and their corresponding tumor types
    image_paths = []
    labels = []

    for tumor_type in os.listdir(data_dir):
        tumor_type_path = data_dir/tumor_type
        if os.path.isdir(tumor_type_path):
            for img_name in os.listdir(tumor_type_path):
                img_path = tumor_type_path / img_name
                # Optional: filter only image files
                if img_path.suffix.lower() in [".png", ".jpg", ".jpeg"]:
                    image_paths.append(str(img_path))
                    labels.append(tumor_type)

    # Create a DataFrame for the dataset
    df = pd.DataFrame({
        "image_path": image_paths,
        "tumor_type": labels
    })

    return df
