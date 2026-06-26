import os
import re
import json
import h5py
import numpy as np
import scipy.io as sio
from PIL import Image
from tqdm import tqdm

# ---------------- CONFIG ----------------
ROOT = "/home/s2989018/master/multilevel_chan/multilevel/datasets/NUSWIDE"
OUT_DIR = "/home/s2989018/master/multilevel_chan/multilevel/datasets/NUSWIDE_djsrh"
IMAGE_DIR = os.path.join(ROOT, "images")

IMG_SIZE = 256

QUERY_JSON = os.path.join(ROOT, "query.json")
TRAIN_JSON = os.path.join(ROOT, "train.json")
DATABASE_JSON = os.path.join(ROOT, "database.json")
VOCAB_JSON = os.path.join(ROOT, "vocab.json")
CONCEPTS_JSON = os.path.join(ROOT, "concepts.json")

LABEL_OUT = os.path.join(OUT_DIR, "nuswide-lall.mat")
TXT_OUT = os.path.join(OUT_DIR, "nuswide-yall.mat")
IMG_OUT = os.path.join(OUT_DIR, "nuswide-iall.mat")

TRAIN_INDEX_OUT = os.path.join(OUT_DIR, "nuswide_train_index.npy")
QUERY_INDEX_OUT = os.path.join(OUT_DIR, "nuswide_query_index.npy")
DATABASE_INDEX_OUT = os.path.join(OUT_DIR, "nuswide_database_index.npy")


# ---------------- HELPERS ----------------

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def tokenize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return text.split()


def get_token_to_id(vocab_obj):
    if isinstance(vocab_obj, dict) and "token_to_id" in vocab_obj:
        return vocab_obj["token_to_id"]
    return vocab_obj


def get_image_path(sample):
    if "image_path" in sample and sample["image_path"]:
        path = sample["image_path"]

        if os.path.isabs(path):
            return path

        if os.path.exists(path):
            return path

    return os.path.join(IMAGE_DIR, f"{sample['photo_id']}.jpg")


def load_image_for_djsrh(path):
    img = Image.open(path).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE))
    arr = np.asarray(img, dtype=np.uint8)

    # datasets.py later does:
    # Image.fromarray(np.transpose(img, (2, 1, 0)))
    # so store as C x W x H.
    arr = np.transpose(arr, (2, 1, 0))

    return arr


def make_bow(sample, token_to_id):
    vec = np.zeros(len(token_to_id), dtype=np.float32)

    if "tags" in sample and sample["tags"]:
        tokens = sample["tags"]
    else:
        tokens = tokenize(sample.get("text", ""))

    for token in tokens:
        if token in token_to_id:
            vec[token_to_id[token]] = 1.0
        elif "[UNK]" in token_to_id:
            vec[token_to_id["[UNK]"]] = 1.0

    if "[PAD]" in token_to_id:
        vec[token_to_id["[PAD]"]] = 0.0

    return vec


def make_label(sample, concepts):
    if "label" in sample:
        return np.asarray(sample["label"], dtype=np.float32)

    concept_to_id = {c: i for i, c in enumerate(concepts)}
    y = np.zeros(len(concepts), dtype=np.float32)

    for label in sample.get("labels", []):
        if label in concept_to_id:
            y[concept_to_id[label]] = 1.0

    return y


def sample_key(sample):
    return str(sample["photo_id"])


# ---------------- MAIN ----------------

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading MLHE NUS-WIDE files...")

    query = load_json(QUERY_JSON)
    train = load_json(TRAIN_JSON)
    database = load_json(DATABASE_JSON)
    vocab_obj = load_json(VOCAB_JSON)
    concepts = load_json(CONCEPTS_JSON)

    token_to_id = get_token_to_id(vocab_obj)

    print(f"Query JSON: {len(query)}")
    print(f"Train JSON: {len(train)}")
    print(f"Database JSON: {len(database)}")
    print(f"Concepts: {len(concepts)}")
    print(f"Vocabulary size: {len(token_to_id)}")

    # Build one global sample list.
    # Query and database must not overlap.
    # Train should be subset of database.
    all_samples = []
    key_to_global = {}

    for split_name, split in [("query", query), ("database", database)]:
        for sample in split:
            key = sample_key(sample)
            if key not in key_to_global:
                key_to_global[key] = len(all_samples)
                all_samples.append(sample)

    print(f"\nGlobal samples before filtering: {len(all_samples)}")

    print("\nChecking image paths only, no JPEG decoding...")

    missing_paths = []

    for sample in tqdm(all_samples):
        path = get_image_path(sample)

        if not os.path.exists(path):
            missing_paths.append(path)

    print(f"Missing images: {len(missing_paths)}")

    if missing_paths:
        print("First missing paths:")
        for path in missing_paths[:10]:
            print(" ", path)

    assert len(missing_paths) == 0, "Some images are missing. Stop before creating DJSRH files."

    valid_samples = all_samples
    old_to_new = {i: i for i in range(len(all_samples))}

    print(f"Valid samples: {len(valid_samples)}")
    print("Image verification skipped because images were already validated during download.")
    def map_indices(split):
        indices = []

        for sample in split:
            key = sample_key(sample)

            if key not in key_to_global:
                continue

            old_idx = key_to_global[key]

            if old_idx in old_to_new:
                indices.append(old_to_new[old_idx])

        return np.asarray(indices, dtype=np.int64)

    indexTest = map_indices(query)
    indexDatabase = map_indices(database)
    indexTrain = map_indices(train)

    print("\nFinal split sizes:")
    print(f"Query/Test: {len(indexTest)}")
    print(f"Database: {len(indexDatabase)}")
    print(f"Train: {len(indexTrain)}")

    assert len(set(indexTest) & set(indexDatabase)) == 0, "Query and database overlap."
    assert set(indexTrain).issubset(set(indexDatabase)), "Train is not subset of database."

    num_samples = len(valid_samples)
    num_labels = len(concepts)
    txt_dim = len(token_to_id)

    print("\nCreating LAll label matrix...")
    LAll = np.zeros((num_samples, num_labels), dtype=np.float32)

    for i, sample in enumerate(tqdm(valid_samples)):
        LAll[i] = make_label(sample, concepts)

    print("Creating YAll text matrix...")
    # DJSRH NUSWIDE loader expects h5py TXT with transpose:
    # txt_set = np.array(txt_file['YAll']).transpose()
    # so save YAll as vocab_size x N.
    YAll = np.zeros((txt_dim, num_samples), dtype=np.float32)

    for i, sample in enumerate(tqdm(valid_samples)):
        YAll[:, i] = make_bow(sample, token_to_id)

    print("Creating IAll image matrix...")
    IAll_shape = (num_samples, 3, IMG_SIZE, IMG_SIZE)

    with h5py.File(IMG_OUT, "w") as f:
        dset = f.create_dataset(
            "IAll",
            shape=IAll_shape,
            dtype=np.uint8
        )

        for i, sample in enumerate(tqdm(valid_samples)):
            path = get_image_path(sample)
            dset[i] = load_image_for_djsrh(path)

    print("Saving LAll...")
    sio.savemat(LABEL_OUT, {"LAll": LAll})

    print("Saving YAll...")
    with h5py.File(TXT_OUT, "w") as f:
        f.create_dataset("YAll", data=YAll)

    print("Saving split indices...")
    np.save(TRAIN_INDEX_OUT, indexTrain)
    np.save(QUERY_INDEX_OUT, indexTest)
    np.save(DATABASE_INDEX_OUT, indexDatabase)

    print("\nDone.")
    print(f"LAll: {LABEL_OUT}")
    print(f"YAll: {TXT_OUT}")
    print(f"IAll: {IMG_OUT}")
    print(f"Train indices: {TRAIN_INDEX_OUT}")
    print(f"Query indices: {QUERY_INDEX_OUT}")
    print(f"Database indices: {DATABASE_INDEX_OUT}")


if __name__ == "__main__":
    main()