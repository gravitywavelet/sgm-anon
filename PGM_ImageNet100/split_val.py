import os
import shutil
import random

src_root = "/root/autodl-tmp/imagenet_small/train"
val_root = "/root/autodl-tmp/imagenet_small/val"

split_ratio = 0.1  # 10% for validation
random.seed(42)

os.makedirs(val_root, exist_ok=True)

for cls in os.listdir(src_root):
    cls_path = os.path.join(src_root, cls)
    if not os.path.isdir(cls_path):
        continue

    images = os.listdir(cls_path)
    random.shuffle(images)

    split_idx = int(len(images) * split_ratio)
    val_imgs = images[:split_idx]

    val_cls_path = os.path.join(val_root, cls)
    os.makedirs(val_cls_path, exist_ok=True)

    for img in val_imgs:
        src = os.path.join(cls_path, img)
        dst = os.path.join(val_cls_path, img)
        shutil.move(src, dst)

print("Done: validation split created.")