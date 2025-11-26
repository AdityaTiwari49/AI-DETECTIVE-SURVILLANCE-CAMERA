import os, random, shutil, glob

# Auto-detect source images folder
CANDIDATES = ["data/images/images", "data/images", "data/images/train/images"]
SRC = None
for c in CANDIDATES:
    if os.path.isdir(c):
        SRC = c
        break

if SRC is None:
    raise SystemExit("No images folder found under data/images. Run 'Get-ChildItem -Recurse .\\data' and paste output to me.")

# gather images
imgs = sorted(glob.glob(os.path.join(SRC, "*.jpg")) + glob.glob(os.path.join(SRC, "*.png")) + glob.glob(os.path.join(SRC, "*.jpeg")))
if len(imgs) == 0:
    raise SystemExit(f"No images found in {SRC}.")

random.seed(42)
random.shuffle(imgs)

val_count = max(int(0.15 * len(imgs)), 50)
test_count = max(int(0.10 * len(imgs)), 30)

val = imgs[:val_count]
test = imgs[val_count: val_count + test_count]
train = imgs[val_count + test_count:]

def ensure(path):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)

for p in ["data/train/images","data/train/labels","data/val/images","data/val/labels","data/test/images","data/test/labels"]:
    ensure(p)

def copy_pairs(image_list, dest_images, dest_labels):
    for img in image_list:
        base = os.path.basename(img)
        name = os.path.splitext(base)[0]
        # possible label locations to try (normalized)
        label_paths = [
            os.path.join(os.path.dirname(img), "..", "labels", name + ".txt"),
            os.path.join(os.path.dirname(img), name + ".txt"),
            os.path.join("data","images","labels", name + ".txt"),
            os.path.join("data","labels", name + ".txt")
        ]
        shutil.copy(img, os.path.join(dest_images, base))
        found = False
        for lp in label_paths:
            lp = os.path.normpath(lp)
            if os.path.exists(lp):
                shutil.copy(lp, os.path.join(dest_labels, name + ".txt"))
                found = True
                break
        if not found:
            # create empty label file if no annotation found
            open(os.path.join(dest_labels, name + ".txt"), "w").close()

copy_pairs(train, "data/train/images", "data/train/labels")
copy_pairs(val, "data/val/images", "data/val/labels")
copy_pairs(test, "data/test/images", "data/test/labels")

print("Split created from source:", SRC)
print(" train:", len(os.listdir("data/train/images")))
print(" val:  ", len(os.listdir("data/val/images")))
print(" test: ", len(os.listdir("data/test/images")))
