import os
import hashlib
import statistics
import pandas as pd
from PIL import Image

data_folder = "data"
class_names = ["AiArtData", "RealArt"]

rows = []
broken_files = []

for class_name in class_names:
    class_folder = os.path.join(data_folder, class_name)

    for root, folders, files in os.walk(class_folder):
        for file_name in files:
            image_path = os.path.join(root, file_name)
            file_size = os.path.getsize(image_path)

            opened_file = open(image_path, "rb")
            file_bytes = opened_file.read()
            opened_file.close()
            file_hash = hashlib.sha256(file_bytes).hexdigest()

            try:
                picture = Image.open(image_path)
                picture.load()
                width, height = picture.size
                colour_mode = picture.mode
                image_format = picture.format
                picture.close()
                readable = True
            except Exception:
                width = 0
                height = 0
                colour_mode = "not readable"
                image_format = "not readable"
                readable = False
                broken_files.append(image_path)

            rows.append({
                "class": class_name,
                "file_name": file_name,
                "path": image_path,
                "size_bytes": file_size,
                "size_kb": round(file_size / 1024, 2),
                "width": width,
                "height": height,
                "colour_mode": colour_mode,
                "format": image_format,
                "sha256": file_hash,
                "readable": readable
            })

table = pd.DataFrame(rows)
table.to_csv("audit.csv", index=False)

hash_groups = {}
for row in rows:
    file_hash = row["sha256"]
    if file_hash not in hash_groups:
        hash_groups[file_hash] = []
    hash_groups[file_hash].append(row["path"])

duplicate_rows = []
duplicate_groups = 0
extra_copies = 0

for file_hash in hash_groups:
    paths = hash_groups[file_hash]
    if len(paths) > 1:
        duplicate_groups = duplicate_groups + 1
        extra_copies = extra_copies + len(paths) - 1
        for path in paths:
            duplicate_rows.append({"sha256": file_hash, "path": path})

pd.DataFrame(duplicate_rows).to_csv("duplicates.csv", index=False)
pd.DataFrame({"broken_file": broken_files}).to_csv("corrupt.csv", index=False)

print("total files:", len(rows))
print("broken files:", len(broken_files))
for path in broken_files:
    print("   ", path)

print("duplicate groups:", duplicate_groups)
print("extra copies:", extra_copies)
print()

for class_name in class_names:
    widths = []
    heights = []
    sizes = []

    for row in rows:
        if row["class"] == class_name and row["readable"]:
            widths.append(row["width"])
            heights.append(row["height"])
            sizes.append(row["size_bytes"])

    print(class_name)
    print("  images:", len(widths))
    if len(widths) > 0:
        print("  width  min/median/max:", min(widths), statistics.median(widths), max(widths))
        print("  height min/median/max:", min(heights), statistics.median(heights), max(heights))
        print("  size   min/median/max:", min(sizes), statistics.median(sizes), max(sizes))
    print()

print("colour modes")
print(table["colour_mode"].value_counts())
print()
print("formats")
print(table["format"].value_counts())