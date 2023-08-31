import os
import shutil
import requests
import tarfile

from tqdm import tqdm


def create_directories(directories: list[str]):
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def download_and_extract_dataset(url: str, target_folder: str):
    fname = "dataset.tar.gz"
    try:
        print("Downloading the dataset...")
        response = requests.get(url, stream=True)
        total = int(response.headers.get("content-length", 0))

        with open(fname, "wb") as f, tqdm(
            desc=fname,
            total=total,
            unit="iB",
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                bar.update(size)

        print("Extracting the dataset...")
        with tarfile.open(fname, "r:gz" if os.name != "nt" else "r:") as tar:
            tar.extractall(target_folder)
        return True
    except Exception as e:
        print("Error while downloading and extracting the dataset:", e)
        return False
    finally:
        print("Cleaning up...")
        if os.path.exists(fname):
            os.remove(fname)


def move_images_to_img_dir(img_dir_path: str):
    extracted_folder_path = os.path.join("datasets/images", "images")
    try:
        print("Moving images to images/ directory...")
        for item in os.listdir(extracted_folder_path):
            src = os.path.join(extracted_folder_path, item)
            dst = os.path.join(img_dir_path, item)
            shutil.move(src, dst)

        print("Successfully moved images to images/ directory.")
        return True
    except Exception as e:
        print("Error while moving images to images/ directory:", e)
        return False
    finally:
        print("Cleaning up...")
        if os.path.exists(extracted_folder_path):
            os.rmdir(extracted_folder_path)


def main():
    directories_to_create = [
        "datasets/numpy/images",
        "datasets/numpy/labels",
        "datasets/images",
        "artifacts/checkpoints",
        "artifacts/csv_logs",
        "artifacts/metrics",
        "artifacts/plots",
        "artifacts/models",
        "artifacts/model_histories",
        "artifacts/model_summaries",
    ]

    create_directories(directories_to_create)

    img_dir_path = "datasets/images"
    if not os.path.exists(img_dir_path) or not os.listdir(img_dir_path):
        dataset_url = "http://aisdatasets.informatik.uni-freiburg.de/freiburg_groceries_dataset/freiburg_groceries_dataset.tar.gz"
        resp = download_and_extract_dataset(dataset_url, "datasets/images")
        if resp:
            resp = move_images_to_img_dir(img_dir_path)
    else:
        print("datasets/images directory already exists and isn't empty.")


if __name__ == "__main__":
    main()
