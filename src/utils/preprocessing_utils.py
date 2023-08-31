import os

import numpy as np

from sklearn import preprocessing, model_selection

from tensorflow import keras

RANDOM_SEED = 42

DATASETS_DIR = "../../datasets"

IMG_DIR = f"{DATASETS_DIR}/images"

NUMPY_DIR = f"{DATASETS_DIR}/numpy"

NUMPY_IMG_DIR = f"{NUMPY_DIR}/images.npy"
NUMPY_LABEL_DIR = f"{NUMPY_DIR}/labels.npy"


class PreprocessingUtils:
    def __init__(self):
        self.label_names: list[str] = os.listdir(IMG_DIR)
        self.images, self.labels, self.num_classes = self.__load_images_and_labels()
        self.labels = self.__preprocess_labels()

        (
            self.X_train,
            self.X_test,
            self.X_val,
            self.y_train,
            self.y_test,
            self.y_val,
        ) = self.__train_test_val_split()

    def __preprocess_and_save_images(self):
        if os.path.exists(NUMPY_IMG_DIR) and os.path.exists(NUMPY_LABEL_DIR):
            print("Files already exist, skipping...")
            return
        else:
            print("Files don't exist, creating...")

        categories = os.listdir(IMG_DIR)

        images = []
        labels = []

        for category in categories:
            image_files = os.listdir(os.path.join(IMG_DIR, category))

            for image_file in image_files:
                img = keras.preprocessing.image.load_img(
                    os.path.join(IMG_DIR, category, image_file), target_size=(256, 256)
                )
                img = keras.preprocessing.image.img_to_array(img)
                img = img / 255.0

                images.append(img)
                labels.append(category)

        np.save(NUMPY_IMG_DIR, np.array(images))
        np.save(NUMPY_LABEL_DIR, np.array(labels))

        print("Images transformed and saved successfully!")

    def __load_images_and_labels(self) -> tuple[np.ndarray, np.ndarray, int]:
        self.__preprocess_and_save_images()

        print("Loading images and labels...")
        images = np.load(NUMPY_IMG_DIR)
        labels = np.load(NUMPY_LABEL_DIR)
        print("Images and labels loaded successfully!")

        num_classes = len(np.unique(labels))

        return images, labels, num_classes

    def __preprocess_labels(self):
        print("Preprocessing labels...")
        le = preprocessing.LabelEncoder()

        labels = le.fit_transform(self.labels)

        labels = keras.utils.to_categorical(labels, self.num_classes)
        print("Labels preprocessed successfully!")

        return labels

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

        return X_train, X_test, X_val, y_train, y_test, y_val
