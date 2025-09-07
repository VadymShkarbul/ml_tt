import os
import zipfile
import shutil
import random
from pathlib import Path
from typing import List, Tuple

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}

ROOT = Path(__file__).resolve().parent
ZIP_PATH = ROOT / 'files.zip'
DATA_ROOT = ROOT / 'data'
RAW_DIR = DATA_ROOT / 'raw'
SPLIT_DIR = DATA_ROOT / 'split'

# Label mapping per guide:
# filenames containing "original" -> class 'screen' (label 1)
# all others -> class 'real' (label 0)
CLASS_DIRS = {
    0: 'real',
    1: 'screen',
}


def unzip_dataset(zip_path: Path, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(out_dir)


def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTS


def collect_images(root: Path) -> List[Path]:
    files: List[Path] = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            p = Path(dirpath) / fn
            if is_image(p):
                files.append(p)
    return files


def infer_label_from_filename(path: Path) -> int:
    name = path.name.lower()
    # Per guide: "original" => screen (1)
    if 'original' in name:
        return 1
    return 0


def train_val_test_split(items: List[Path], ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15), seed: int = 42):
    assert abs(sum(ratios) - 1.0) < 1e-6, 'Ratios must sum to 1.0'
    rng = random.Random(seed)
    shuffled = items[:]
    rng.shuffle(shuffled)
    n = len(shuffled)
    n_train = int(n * ratios[0])
    n_val = int(n * ratios[1])
    train = shuffled[:n_train]
    val = shuffled[n_train:n_train + n_val]
    test = shuffled[n_train + n_val:]
    return train, val, test


def prepare_split_dir(split_dir: Path):
    # Clean existing split dir to avoid stale files
    if split_dir.exists():
        shutil.rmtree(split_dir)
    for split in ('train', 'val', 'test'):
        for cls in CLASS_DIRS.values():
            (split_dir / split / cls).mkdir(parents=True, exist_ok=True)


def copy_with_structure(files: List[Tuple[Path, int]], split_name: str, split_dir: Path):
    for src_path, label in files:
        cls_name = CLASS_DIRS[label]
        dst_dir = split_dir / split_name / cls_name
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst_path = dst_dir / src_path.name
        # If duplicate names, avoid overwriting by adding an index
        if dst_path.exists():
            stem, ext = src_path.stem, src_path.suffix
            i = 1
            while True:
                candidate = dst_dir / f"{stem}_{i}{ext}"
                if not candidate.exists():
                    dst_path = candidate
                    break
                i += 1
        shutil.copy2(src_path, dst_path)


def main():
    if not ZIP_PATH.exists():
        raise FileNotFoundError(f"Dataset archive not found at {ZIP_PATH}")

    # 1) Unzip into data/raw
    print(f"Unzipping {ZIP_PATH} -> {RAW_DIR}")
    if RAW_DIR.exists() and any(RAW_DIR.iterdir()):
        print("Raw directory already populated, skipping unzip.")
    else:
        unzip_dataset(ZIP_PATH, RAW_DIR)

    # 2) Collect images and infer labels per filename rule
    print("Collecting images...")
    images = collect_images(RAW_DIR)
    if not images:
        raise RuntimeError(f"No images found under {RAW_DIR}")
    labeled = [(p, infer_label_from_filename(p)) for p in images]

    # 3) Split
    print(f"Total images: {len(labeled)}")
    train_files, val_files, test_files = train_val_test_split([p for p, _ in labeled])
    # Re-attach labels after split
    label_map = {p: infer_label_from_filename(p) for p, _ in labeled}
    train_labeled = [(p, label_map[p]) for p in train_files]
    val_labeled = [(p, label_map[p]) for p in val_files]
    test_labeled = [(p, label_map[p]) for p in test_files]

    # 4) Create a split structure and copy files
    print(f"Preparing split directories in {SPLIT_DIR}")
    prepare_split_dir(SPLIT_DIR)
    print("Copying train files...")
    copy_with_structure(train_labeled, 'train', SPLIT_DIR)
    print("Copying val files...")
    copy_with_structure(val_labeled, 'val', SPLIT_DIR)
    print("Copying test files...")
    copy_with_structure(test_labeled, 'test', SPLIT_DIR)

    print("Data preparation completed.")


if __name__ == '__main__':
    main()
