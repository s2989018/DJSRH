import os
import random
import numpy as np
import scipy.io as sio
import h5py
from PIL import Image
from tqdm import tqdm
from collections import Counter

# ---------------- CONFIG ----------------

DATA_DIR = "mirflickr"
IMG_DIR = DATA_DIR
TAG_DIR = os.path.join(DATA_DIR, "meta", "tags")
CLASS_DIR = "mirflickr25k_annotations_v080"

OUT_DIR = "./datasets/MIRFlickr_djsrh"
os.makedirs(OUT_DIR, exist_ok=True)

SEED = 42
MIN_TAG_FREQ = 20
IMG_SIZE = 256

random.seed(SEED)
np.random.seed(SEED)

# -------------------------------
# 1. LOAD TAGS
# -------------------------------

print("Reading tags...")

image_tags = {}

for tag_file in sorted(os.listdir(TAG_DIR)):
    if not tag_file.endswith(".txt"):
        continue

    idx = ''.join(filter(str.isdigit, tag_file))
    path = os.path.join(TAG_DIR, tag_file)

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        tags = [t.strip() for t in f if t.strip()]

    if tags:
        image_tags[idx] = tags

# -------------------------------
# 2. FILTER TAGS BY FREQUENCY
# -------------------------------

print("Filtering tags by frequency...")

tag_counts = Counter()

for tags in image_tags.values():
    tag_counts.update(set(tags))

frequent_tags = sorted([
    tag for tag, count in tag_counts.items()
    if count >= MIN_TAG_FREQ
])

tag_idx = {tag: i for i, tag in enumerate(frequent_tags)}
DIM_TXT = len(tag_idx)

print("Text feature dimension:", DIM_TXT)

# -------------------------------
# 3. LOAD CLASS ANNOTATIONS
# -------------------------------

print("Loading class annotations...")

class_files = sorted([
    f for f in os.listdir(CLASS_DIR)
    if os.path.isfile(os.path.join(CLASS_DIR, f))
    and f.lower().endswith(".txt")
    and not f.lower().endswith("_r1.txt")
    and f.lower() != "readme.txt"
])

DIM_LABEL = len(class_files)

print("Label dimension:", DIM_LABEL)
print("Class files:")
for f in class_files:
    print(" ", f)

class_idx = {c: i for i, c in enumerate(class_files)}

image_labels = {}

for cname in class_files:
    path = os.path.join(CLASS_DIR, cname)

    with open(path, "r") as f:
        for line in f:
            idx = line.strip()

            if not idx.isdigit():
                continue

            if idx not in image_labels:
                image_labels[idx] = np.zeros(DIM_LABEL, dtype=np.float32)

            image_labels[idx][class_idx[cname]] = 1.0

# -------------------------------
# 4. COLLECT VALID SAMPLES
# -------------------------------

print("Collecting valid samples...")

samples = []

for idx, tags in image_tags.items():
    if idx not in image_labels:
        continue

    filtered_tags = [t for t in tags if t in tag_idx]

    if len(filtered_tags) == 0:
        continue

    img_path = os.path.join(IMG_DIR, f"im{idx}.jpg")

    if not os.path.exists(img_path):
        continue

    if image_labels[idx].sum() == 0:
        continue

    samples.append({
        "id": idx,
        "image_path": img_path,
        "tags": filtered_tags,
        "label": image_labels[idx]
    })

samples = sorted(samples, key=lambda x: int(x["id"]))

print("Total valid samples:", len(samples))

# -------------------------------
# 5. BUILD TEXT AND LABEL MATRICES
# -------------------------------

def build_tag_vec(tags):
    vec = np.zeros(DIM_TXT, dtype=np.float32)
    for tag in tags:
        if tag in tag_idx:
            vec[tag_idx[tag]] = 1.0
    return vec

YAll = []
LAll = []
IDs = []

for s in samples:
    YAll.append(build_tag_vec(s["tags"]))
    LAll.append(s["label"])
    IDs.append(s["id"])

YAll = np.array(YAll, dtype=np.float32)
LAll = np.array(LAll, dtype=np.float32)
IDs = np.array(IDs)

print("YAll shape:", YAll.shape)
print("LAll shape:", LAll.shape)

# -------------------------------
# 6. SAVE LABEL AND TEXT MAT FILES
# -------------------------------

label_path = os.path.join(OUT_DIR, "mirflickr25k-lall.mat")
txt_path = os.path.join(OUT_DIR, "mirflickr25k-yall.mat")

print("Saving labels:", label_path)
sio.savemat(label_path, {
    "LAll": LAll,
    "IDAll": IDs
})

print("Saving text features:", txt_path)
sio.savemat(txt_path, {
    "YAll": YAll,
    "IDAll": IDs
})

# -------------------------------
# 7. SAVE IMAGE HDF5 MAT FILE
# -------------------------------

img_mat_path = os.path.join(OUT_DIR, "mirflickr25k-iall.mat")

print("Saving images:", img_mat_path)

with h5py.File(img_mat_path, "w") as f:
    IAll = f.create_dataset(
        "IAll",
        shape=(len(samples), 3, IMG_SIZE, IMG_SIZE),
        dtype=np.uint8
    )

    id_ds = f.create_dataset(
        "IDAll",
        data=IDs.astype("S")
    )

    for i, s in enumerate(tqdm(samples)):
        img = Image.open(s["image_path"]).convert("RGB")
        img = img.resize((IMG_SIZE, IMG_SIZE))

        arr = np.array(img, dtype=np.uint8)          # H, W, 3
        arr = np.transpose(arr, (2, 1, 0))           # 3, W, H

        IAll[i] = arr

print("Finished.")
print("Output files:")
print(" ", label_path)
print(" ", txt_path)
print(" ", img_mat_path)