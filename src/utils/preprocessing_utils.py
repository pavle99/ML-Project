import os

import numpy as np

from sklearn import preprocessing, model_selection

from tensorflow import keras

RANDOM_SEED = 42

DATASETS_DIR = "../../datasets"

IMAGES_DIR = f"{DATASETS_DIR}/images"

NUMPY_DIR = f"{DATASETS_DIR}/numpy"

NUMPY_IMAGES_DIR = f"{NUMPY_DIR}/images"
NUMPY_LABELS_DIR = f"{NUMPY_DIR}/labels"


class PreprocessingUtils:
    def __init__(self, force_split: bool = False):
        self.label_names: list[str] = os.listdir(IMAGES_DIR)
        self.num_classes = len(self.label_names)

        self.images, self.labels = self.__load_images_and_labels()
        self.labels = self.__preprocess_labels()

        (
            self.X_train,
            self.X_test,
            self.X_val,
            self.y_train,
            self.y_test,
            self.y_val,
        ) = (
            self.__train_test_val_split() if force_split else self.__load_split_data()
        )

    def __preprocess_and_save_images(self):
        if os.path.exists(f"{NUMPY_IMAGES_DIR}/all.npy") and os.path.exists(
            f"{NUMPY_LABELS_DIR}/all.npy"
        ):
            print("Files already exist, skipping...")
            return
        else:
            print("Files don't exist, creating...")

        images = []
        labels = []

        for category in self.label_names:
            image_files = os.listdir(os.path.join(IMAGES_DIR, category))

            for image_file in image_files:
                img = keras.preprocessing.image.load_img(
                    os.path.join(IMAGES_DIR, category, image_file),
                    target_size=(256, 256),
                )
                img = keras.preprocessing.image.img_to_array(img)
                img = img / 255.0

                images.append(img)
                labels.append(category)

        np.save(f"{NUMPY_IMAGES_DIR}/all.npy", np.array(images))
        np.save(f"{NUMPY_LABELS_DIR}/all.npy", np.array(labels))

        print("Images transformed and saved successfully!")

    def __load_images_and_labels(self) -> tuple[np.ndarray, np.ndarray]:
        self.__preprocess_and_save_images()

        print("Loading images and labels...")
        images = np.load(f"{NUMPY_IMAGES_DIR}/all.npy")
        labels = np.load(f"{NUMPY_LABELS_DIR}/all.npy")
        print("Images and labels loaded successfully!")

        return images, labels

    def __preprocess_labels(self):
        print("Preprocessing labels...")
        le = preprocessing.LabelEncoder()

        labels = le.fit_transform(self.labels)

        labels = keras.utils.to_categorical(labels, self.num_classes)
        print("Labels preprocessed successfully!")

        return labels

    def __load_split_data(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        if not all(
            [
                os.path.exists(f"{NUMPY_IMAGES_DIR}/{split}.npy")
                and os.path.exists(f"{NUMPY_LABELS_DIR}/{split}.npy")
                for split in ["train", "test", "val"]
            ]
        ):
            print("Files don't exist, splitting...")
            return self.__train_test_val_split()

        print("Loading split data...")
        X_train = np.load(f"{NUMPY_IMAGES_DIR}/train.npy")
        X_test = np.load(f"{NUMPY_IMAGES_DIR}/test.npy")
        X_val = np.load(f"{NUMPY_IMAGES_DIR}/val.npy")

        y_train = np.load(f"{NUMPY_LABELS_DIR}/train.npy")
        y_test = np.load(f"{NUMPY_LABELS_DIR}/test.npy")
        y_val = np.load(f"{NUMPY_LABELS_DIR}/val.npy")
        print("Split data loaded successfully!")

        return X_train, X_test, X_val, y_train, y_test, y_val

    def __train_test_val_split(
        self, test_size: float = 1 / 8, val_size: float = 1 / 20
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        print("Splitting data into train, test and validation sets...")
        X_train_val, X_test, y_train_val, y_test = model_selection.train_test_split(
            self.images, self.labels, test_size=test_size, random_state=RANDOM_SEED
        )

        X_train, X_val, y_train, y_val = model_selection.train_test_split(
            X_train_val, y_train_val, test_size=val_size, random_state=RANDOM_SEED
        )
        print("Data split successfully!")

        print("Saving split data...")
        np.save(f"{NUMPY_IMAGES_DIR}/train.npy", X_train)
        np.save(f"{NUMPY_IMAGES_DIR}/test.npy", X_test)
        np.save(f"{NUMPY_IMAGES_DIR}/val.npy", X_val)

        np.save(f"{NUMPY_LABELS_DIR}/train.npy", y_train)
        np.save(f"{NUMPY_LABELS_DIR}/test.npy", y_test)
        np.save(f"{NUMPY_LABELS_DIR}/val.npy", y_val)
        print("Split data saved successfully!")

        return X_train, X_test, X_val, y_train, y_test, y_val
