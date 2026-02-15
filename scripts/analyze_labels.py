import os
from collections import Counter

label_dir = r"data\TILDA_yolo\labels"
class_counts = Counter()
total_files = 0
empty_files = 0

for filename in os.listdir(label_dir):
    if filename.endswith(".txt"):
        total_files += 1
        filepath = os.path.join(label_dir, filename)
        with open(filepath, 'r') as f:
            lines = f.readlines()
            if not lines:
                empty_files += 1
            for line in lines:
                parts = line.split()
                if parts:
                    class_id = parts[0]
                    class_counts[class_id] += 1

print(f"Total label files: {total_files}")
print(f"Empty label files: {empty_files}")
print(f"Class distribution: {dict(class_counts)}")
