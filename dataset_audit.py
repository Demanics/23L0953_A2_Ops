import os
import hashlib
import statistics
import pandas as pd
from PIL import Image

data_folder = "data"
artifacts_folder = "Artifacts"
class_names = ["AiArtData", "RealArt"]

os.makedirs(artifacts_folder, exist_ok=True)

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
table.to_csv(os.path.join(artifacts_folder, "audit.csv"), index=False)

hash_groups = {}
hash_classes = {}

for row in rows:
    file_hash = row["sha256"]
    if file_hash not in hash_groups:
        hash_groups[file_hash] = []
        hash_classes[file_hash] = []
    hash_groups[file_hash].append(row["path"])
    if row["class"] not in hash_classes[file_hash]:
        hash_classes[file_hash].append(row["class"])

duplicate_rows = []
duplicate_groups = 0
extra_copies = 0
cross_class_duplicates = 0

for file_hash in hash_groups:
    paths = hash_groups[file_hash]
    if len(paths) > 1:
        duplicate_groups = duplicate_groups + 1
        extra_copies = extra_copies + len(paths) - 1
        if len(hash_classes[file_hash]) > 1:
            cross_class_duplicates = cross_class_duplicates + 1
        for path in paths:
            duplicate_rows.append({"sha256": file_hash, "path": path})

pd.DataFrame(duplicate_rows).to_csv(os.path.join(artifacts_folder, "duplicates.csv"), index=False)
pd.DataFrame({"broken_file": broken_files}).to_csv(os.path.join(artifacts_folder, "corrupt.csv"), index=False)

summary_lines = []
summary_lines.append("total files: " + str(len(rows)))
summary_lines.append("broken files: " + str(len(broken_files)))

for path in broken_files:
    summary_lines.append("    " + path)

summary_lines.append("duplicate groups: " + str(duplicate_groups))
summary_lines.append("extra copies: " + str(extra_copies))
summary_lines.append("duplicates across classes: " + str(cross_class_duplicates))
summary_lines.append("total size mb: " + str(round(table["size_bytes"].sum() / 1024 / 1024, 2)))
summary_lines.append("")

for class_name in class_names:
    widths = []
    heights = []
    sizes = []

    for row in rows:
        if row["class"] == class_name and row["readable"]:
            widths.append(row["width"])
            heights.append(row["height"])
            sizes.append(row["size_bytes"])

    summary_lines.append(class_name)
    summary_lines.append("  images: " + str(len(widths)))

    if len(widths) > 0:
        summary_lines.append("  width  min/median/max: " + str(min(widths)) + " " + str(statistics.median(widths)) + " " + str(max(widths)))
        summary_lines.append("  height min/median/max: " + str(min(heights)) + " " + str(statistics.median(heights)) + " " + str(max(heights)))
        summary_lines.append("  size   min/median/max: " + str(min(sizes)) + " " + str(statistics.median(sizes)) + " " + str(max(sizes)))

    summary_lines.append("")

summary_lines.append("colour modes")
summary_lines.append(table["colour_mode"].value_counts().to_string())
summary_lines.append("")
summary_lines.append("formats")
summary_lines.append(table["format"].value_counts().to_string())

for line in summary_lines:
    print(line)

summary_file = open(os.path.join(artifacts_folder, "summary.txt"), "w")
summary_file.write("\n".join(summary_lines))
summary_file.close()

print("saved audit files in " + artifacts_folder)