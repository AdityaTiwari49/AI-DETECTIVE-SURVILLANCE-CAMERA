import glob, os

paths = {
    "train": "data/train/images",
    "val": "data/val/images",
    "test": "data/test/images"
}

nc = 2
errors = 0

for split, path in paths.items():
    imgs = sorted(
        glob.glob(os.path.join(path, "*.jpg")) +
        glob.glob(os.path.join(path, "*.png")) +
        glob.glob(os.path.join(path, "*.jpeg"))
    )
    print(f"{split}: {len(imgs)} images")

    for img in imgs:
        lbl = os.path.splitext(img)[0] + ".txt"

        if not os.path.exists(lbl):
            print("  MISSING LABEL:", lbl)
            errors += 1
            continue

        with open(lbl, "r") as f:
            for ln in f:
                parts = ln.strip().split()
                if len(parts) == 0:
                    continue
                if len(parts) < 5:
                    print("  BAD FORMAT:", lbl, ln.strip())
                    errors += 1
                    break

                try:
                    cls = int(parts[0])
                except:
                    print("  INVALID CLASS:", lbl, ln.strip())
                    errors += 1
                    break

                if cls < 0 or cls >= nc:
                    print("  OUT OF RANGE CLASS:", lbl, cls)
                    errors += 1
                    break

print("Done. errors:", errors)
