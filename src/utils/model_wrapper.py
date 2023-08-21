from typing import Callable, Union

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from sklearn import metrics

from tensorflow import keras

import sys

sys.path.append("..")
from utils.preprocessing_utils import PreprocessingUtils


ARTIFACT_DIR = "../../artifacts"


class ModelWrapper:
    def __init__(self, model_name: str, preprocessing_utils: PreprocessingUtils):
        self.model_name = model_name

        self.preprocessing_utils = preprocessing_utils

        self.model: keras.models.Sequential

        self.history: keras.callbacks.History
        self.loss = 0.0
        self.accuracy = 0.0

    def build_model(self, build_fn: Callable, **kwargs):
        print("Building model...")
        self.model = build_fn(**kwargs)
        print("Model built successfully!")

    def train_evaluate_and_save_model(
        self,
        epochs: int = 100,
        batch_size: int = 500,
        callbacks: list[keras.callbacks.Callback] = [],
    ):
        callbacks.append(keras.callbacks.EarlyStopping(monitor="val_loss", patience=20))
        callbacks.append(
            keras.callbacks.ModelCheckpoint(
                filepath=f"{ARTIFACT_DIR}/checkpoints/checkpoint_{self.model_name}", verbose=1, save_best_only=True
            )
        )
        callbacks.append(keras.callbacks.CSVLogger(f"{ARTIFACT_DIR}/csv_logs/history_{self.model_name}.log"))

        self.history = self.model.fit(
            self.preprocessing_utils.X_train,
            self.preprocessing_utils.y_train,
            validation_data=(self.preprocessing_utils.X_val, self.preprocessing_utils.y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=callbacks,
        )

        self.loss, self.accuracy = self.model.evaluate(self.preprocessing_utils.X_test, self.preprocessing_utils.y_test)

        print(f'Saving model to "artifacts/models/{self.model_name}.h5"...')
        self.model.save(f"{ARTIFACT_DIR}/models/{self.model_name}.h5")
        print("Model saved successfully!")

    def load_model(self, model_path: Union[str, None]):
        if model_path is None:
            model_path = f"{ARTIFACT_DIR}/models/{self.model_name}.h5"

        print(f'Loading model from "{model_path}"...')
        model = keras.models.load_model(model_path)
        if model is None:
            print("Model not found!")
            return

        self.model = model
        print("Model loaded successfully!")

    def plot_accuracy(self):
        plt.plot(self.history.history["accuracy"], label="train accuracy")
        plt.plot(self.history.history["val_accuracy"], label="validation accuracy")
        plt.title("Model accuracy")
        plt.ylabel("Accuracy")
        plt.xlabel("Epoch")
        plt.legend(loc="best")
        plt.savefig(f"{ARTIFACT_DIR}/plots/Accuracy_{self.model_name}")
        plt.show()

    def plot_loss(self):
        plt.plot(self.history.history["loss"], label="train loss")
        plt.plot(self.history.history["val_loss"], label="validation loss")
        plt.title("Model loss")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.legend(loc="best")
        plt.savefig(f"{ARTIFACT_DIR}/plots/Loss_{self.model_name}")
        plt.show()

    def display_confusion_matrix(self):
        y_pred = self.model.predict(self.preprocessing_utils.X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true = np.argmax(self.preprocessing_utils.y_test, axis=1)

        cmap = "viridis"
        cm_plot_labels = [i for i in range(25)]

        cm = metrics.confusion_matrix(y_true=y_true, y_pred=y_pred_classes)
        df_cm = pd.DataFrame(cm, cm_plot_labels, cm_plot_labels)
        sns.set(font_scale=1.1)
        plt.figure(figsize=(15, 10))
        s = sns.heatmap(df_cm, annot=True, cmap=cmap)
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.savefig(f"{ARTIFACT_DIR}/metrics/ConfusionMatrix_{self.model_name}.png")
        plt.show()
