import numpy as np
import os
import shutil
from tqdm import tqdm

TEST_SPLIT_RATIO = 0.2

original_data_dir = "data/segmented_texts"
new_data_dir = "data/youtube_comments"
train_dir = os.path.join(new_data_dir, "train")
test_dir = os.path.join(new_data_dir, "test")

os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)


for label in ["neg", "pos"]:
    old_label_dir = os.path.join(original_data_dir, label)
    label_train_dir = os.path.join(train_dir, label)
    label_test_dir = os.path.join(test_dir, label)

    os.makedirs(label_train_dir, exist_ok=True)
    os.makedirs(label_test_dir, exist_ok=True)

    file_list = np.array(os.listdir(old_label_dir))
    n_samples = len(file_list)
    n_test_samples = int(len(file_list) * 0.2)

    test_idx = np.random.choice(n_samples, n_test_samples, replace=False)
    test_files = file_list[test_idx]
    train_files = file_list[np.delete(np.arange(n_samples), test_idx)]

    for file in tqdm(train_files):
        shutil.copyfile(os.path.join(old_label_dir, file), os.path.join(label_train_dir, file))
    for file in tqdm(test_files):
        shutil.copyfile(os.path.join(old_label_dir, file), os.path.join(label_test_dir, file))
