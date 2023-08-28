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

        self.callbacks: list[keras.callbacks.Callback] = self.init_callbacks()

        self.model: keras.models.Sequential

        self.history: keras.callbacks.History
        self.loss = 0.0
        self.accuracy = 0.0

    def build_model(self, build_fn: Callable, **kwargs):
        print("Building model...")
        self.model = build_fn(**kwargs)
        print("Model built successfully!")

        print(f"Saving model summary to {ARTIFACT_DIR}/model_summaries/summary_{self.model_name}...")
        keras.utils.plot_model(
            self.model,
            to_file=f"{ARTIFACT_DIR}/model_summaries/summary_{self.model_name}.png",
            show_shapes=True,
            show_layer_names=True,
        )
        print("Model summary saved successfully!")

    def init_callbacks(self):
        early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=30)
        checkpointer = keras.callbacks.ModelCheckpoint(
            filepath=f"{ARTIFACT_DIR}/checkpoints/checkpoint_{self.model_name}", verbose=1, save_best_only=True
        )
        csv_logger = keras.callbacks.CSVLogger(f"{ARTIFACT_DIR}/csv_logs/history_{self.model_name}.log")
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6)  # type: ignore

        return [
            early_stopping,
            checkpointer,
            csv_logger,
            reduce_lr,
        ]

    def train_and_save_model(
        self,
        epochs: int = 100,
        batch_size: int = 500,
        callbacks: list[keras.callbacks.Callback] = [],
    ):
        self.callbacks.extend(callbacks)

        self.history = self.model.fit(
            self.preprocessing_utils.X_train,
            self.preprocessing_utils.y_train,
            validation_data=(self.preprocessing_utils.X_val, self.preprocessing_utils.y_val),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=self.callbacks,
        )

        print(f'Saving model to "artifacts/models/{self.model_name}.h5"...')
        self.model.save(f"{ARTIFACT_DIR}/models/{self.model_name}.h5")
        print("Model saved successfully!")

    def load_model(self, model_path: Union[str, None] = None):
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

    def __display_loss_and_accuracy(self):
        self.loss, self.accuracy = self.model.evaluate(self.preprocessing_utils.X_test, self.preprocessing_utils.y_test)

        print(f"Loss: {self.loss:.4f}")
        print(f"Accuracy: {self.accuracy:.4f}")

    def __display_confusion_matrix(self, y_pred: np.ndarray, y_true: np.ndarray):
        cmap = "viridis"
        cm_plot_labels = [i for i in range(25)]

        cm = metrics.confusion_matrix(y_true=y_true, y_pred=y_pred)
        df_cm = pd.DataFrame(cm, cm_plot_labels, cm_plot_labels)
        sns.set(font_scale=1.1)
        plt.figure(figsize=(15, 10))
        s = sns.heatmap(df_cm, annot=True, cmap=cmap)
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.savefig(f"{ARTIFACT_DIR}/metrics/ConfusionMatrix_{self.model_name}.png")
        plt.show()

    def display_evaluation_results(self):
        self.__display_loss_and_accuracy()

        y_pred = self.model.predict(self.preprocessing_utils.X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true = np.argmax(self.preprocessing_utils.y_test, axis=1)

        print(f"Classification Report:\n{metrics.classification_report(y_true, y_pred_classes)}")

        self.__display_confusion_matrix(y_pred_classes, y_true)
