import os
import shutil
import requests
import tarfile


def create_directories(directories: list[str]):
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def download_and_extract_dataset(url: str, target_folder: str):
    fname = "dataset.tar.gz"
    try:
        print("Downloading the dataset...")
        response = requests.get(url, stream=True)

        with open(fname, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

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


def move_images_to_original(original_folder_path):
    extracted_folder_path = os.path.join("datasets/images", "images")
    try:
        print("Moving images to original/ folder...")
        for item in os.listdir(extracted_folder_path):
            src = os.path.join(extracted_folder_path, item)
            dst = os.path.join(original_folder_path, item)
            shutil.move(src, dst)

        print("Successfully moved images to original/ folder.")
        return True
    except Exception as e:
        print("Error while moving images to original folder:", e)
        return False
    finally:
        print("Cleaning up...")
        if os.path.exists(extracted_folder_path):
            os.rmdir(extracted_folder_path)


def main():
    directories_to_create = [
        "datasets/numpy",
        "datasets/images/original",
        "datasets/images/test",
        "datasets/images/train",
        "datasets/images/val",
        "artifacts/checkpoints",
        "artifacts/csv_logs",
        "artifacts/metrics",
        "artifacts/plots",
    ]

    create_directories(directories_to_create)

    original_folder_path = "datasets/images/original"
    if not os.path.exists(original_folder_path) or not os.listdir(original_folder_path):
        dataset_url = (
            "http://aisdatasets.informatik.uni-freiburg.de/freiburg_groceries_dataset/freiburg_groceries_dataset.tar.gz"
        )
        resp = download_and_extract_dataset(dataset_url, "datasets/images")
        if resp:
            resp = move_images_to_original(original_folder_path)
    else:
        print("Dataset 'original' folder already exists and isn't empty.")


if __name__ == "__main__":
    main()
