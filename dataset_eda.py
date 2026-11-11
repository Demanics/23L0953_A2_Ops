import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

artifacts_folder = "Artifacts"
os.makedirs(artifacts_folder, exist_ok=True)

table = pd.read_csv(os.path.join(artifacts_folder, "audit.csv"))
table = table[table["readable"] == True]

class_names = ["AiArtData", "RealArt"]
samples_per_class = 5

figure, axes = plt.subplots(len(class_names), samples_per_class, figsize=(12, 6))

for row_number in range(len(class_names)):
    class_name = class_names[row_number]
    class_rows = table[table["class"] == class_name].head(samples_per_class)

    for column_number in range(samples_per_class):
        image_path = class_rows.iloc[column_number]["path"]
        picture = Image.open(image_path).convert("RGB")
        axes[row_number][column_number].imshow(picture)
        axes[row_number][column_number].set_title(class_name, fontsize=9)
        axes[row_number][column_number].axis("off")

figure.suptitle("Sample images per class")
figure.tight_layout()
figure.savefig(os.path.join(artifacts_folder, "samples.png"), dpi=120)
plt.close(figure)

columns = ["width", "height", "size_kb"]
labels = ["width (pixels)", "height (pixels)", "file size (KB)"]

for number in range(3):
    column_name = columns[number]
    plt.figure(figsize=(7, 4))

    for class_name in class_names:
        values = table[table["class"] == class_name][column_name]
        plt.hist(values, bins=30, alpha=0.5, label=class_name)

    plt.xlabel(labels[number])
    plt.ylabel("number of images")
    plt.title(labels[number] + " per class")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(artifacts_folder, "hist_" + column_name + ".png"), dpi=120)
    plt.close()

print("saved figures in " + artifacts_folder)