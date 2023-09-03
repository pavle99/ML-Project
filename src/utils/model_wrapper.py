from typing import Callable, Literal

import pickle

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
    """
    A wrapper for building, training, evaluating, and visualizing neural network models using the Keras library.

    Attributes
    ----------
    model_name : str
        The unique identifier name for the model.
    preprocessing_utils : PreprocessingUtils
        A utility object for preprocessing data before feeding it into the model.
    callbacks : list[keras.callbacks.Callback]
        A list of callback objects to apply during model training.
    model : keras.models.Sequential
        The actual model architecture.
    history : dict
        The training and validation history of the model.
    loss : float
        The loss value after evaluating the model.
    accuracy : float
        The accuracy score after evaluating the model.

    Methods
    -------
    build_model(build_fn: Callable, **kwargs):
        Build the neural network model using the provided function and save its summary.
    init_callbacks():
        Initialize default callbacks for the model training.
    train_and_save_model_and_history(epochs: int = 100, batch_size: int = 500, callbacks: list[keras.callbacks.Callback] = []):
        Train the model and save it alongside its training history.
    load_model_and_history(model_path: str = "", history_path: str = ""):
        Load the saved model and its training history.
    plot_accuracy():
        Plot and save the training and validation accuracy of the model.
    plot_loss():
        Plot and save the training and validation loss of the model.
    display_evaluation_results():
        Display the model's loss, accuracy, classification report, and confusion matrix.
    display_random_image_prediction():
        Visualize the model's prediction for a randomly chosen image from the test set.
    display_all_image_predictions_for_label(label_name: str, limit: int = 5):
        Visualize the model's predictions for all images of a specific label from the test set.

    Notes
    -----
    The class makes use of many popular Python libraries like Keras, Matplotlib, Seaborn, etc. for various purposes like model building, plotting, etc.

    Examples
    --------
    Initialize the PreprocessingUtils and ModelWrapper objects:

    >>> preprocessing_utils = PreprocessingUtils()
    >>> model_wrapper = ModelWrapper("ModelName", preprocessing_utils=preprocessing_utils)

    Define a function to build the model:

    >>> def build_model_fn():
    ...     ...

    Build the model, train it, and save it alongside its history:

    >>> model_wrapper.build_model(build_fn=build_model_fn, model_name=model_wrapper.model_name)
    >>> model_wrapper.train_and_save_model_and_history()

    Load the model and its history:
    >>> model_wrapper.load_model_and_history()

    Plot the training and validation accuracy and loss:
    >>> model_wrapper.plot_accuracy()
    >>> model_wrapper.plot_loss()

    Display the model's loss, accuracy, classification report, and confusion matrix:

    >>> model_wrapper.display_evaluation_results()
    """

    def __init__(self, model_name: str, preprocessing_utils: PreprocessingUtils):
        """
        Initialize the ModelWrapper.

        Parameters
        ----------
        model_name : str
            Name of the model, used for saving and logging purposes.
        preprocessing_utils : PreprocessingUtils
            An instance of PreprocessingUtils containing the data to be used.
        """

        self.model_name = model_name

        self.preprocessing_utils = preprocessing_utils

        self.callbacks: list[keras.callbacks.Callback] = self.__init_callbacks()

        self.model: keras.models.Sequential

        self.history: dict
        self.loss = 0.0
        self.accuracy = 0.0

    def build_model(self, build_fn: Callable, **kwargs):
        """
        Build the model using the provided function and save its summary.

        Parameters
        ----------
        build_fn : Callable
            A function that is used to build the model.
        **kwargs : dict
            Arguments to be passed to the `build_fn` function.

        Notes
        -----
        The model summary is saved to "artifacts/model_summaries/summary_{self.model_name}.png".
        """

        print("Building model...")
        self.model = build_fn(**kwargs)
        print("Model built successfully!")

        print(
            f"Saving model summary to {ARTIFACT_DIR}/model_summaries/summary_{self.model_name}..."
        )
        keras.utils.plot_model(
            self.model,
            to_file=f"{ARTIFACT_DIR}/model_summaries/summary_{self.model_name}.png",
            show_shapes=True,
            show_layer_names=True,
        )
        print("Model summary saved successfully!")

    def __init_callbacks(self):
        """
        Initialize default callbacks for the model training.

        Returns
        -------
        list[keras.callbacks.Callback]
            A list of callback objects to apply during model training.

        Notes
        -----
        The default callbacks are:
            - EarlyStopping
            - ModelCheckpoint
            - CSVLogger
            - ReduceLROnPlateau
        """
        early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=30)
        checkpointer = keras.callbacks.ModelCheckpoint(
            filepath=f"{ARTIFACT_DIR}/checkpoints/checkpoint_{self.model_name}",
            verbose=1,
            save_best_only=True,
        )
        csv_logger = keras.callbacks.CSVLogger(
            f"{ARTIFACT_DIR}/csv_logs/history_{self.model_name}.log"
        )
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6)  # type: ignore

        return [
            early_stopping,
            checkpointer,
            csv_logger,
            reduce_lr,
        ]

    def train_and_save_model_and_history(
        self,
        epochs: int = 100,
        batch_size: int = 500,
        callbacks: list[keras.callbacks.Callback] = [],
    ):
        """
        Train the model and save the model and its training history.

        Parameters
        ----------
        epochs : int, optional
            Number of epochs for training, by default 100.
        batch_size : int, optional
            Batch size for training, by default 500.
        callbacks : list[keras.callbacks.Callback], optional
            List of additional callbacks to append, by default empty.

        Notes
        -----
        The model is saved to "artifacts/models/{self.model_name}.h5".
        The history is saved to "artifacts/model_histories/history_{self.model_name}".
        """

        self.callbacks.extend(callbacks)

        history = self.model.fit(
            self.preprocessing_utils.X_train,
            self.preprocessing_utils.y_train,
            validation_data=(
                self.preprocessing_utils.X_val,
                self.preprocessing_utils.y_val,
            ),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=self.callbacks,
        )

        print(f'Saving model to "artifacts/models/{self.model_name}.h5"...')
        self.model.save(f"{ARTIFACT_DIR}/models/{self.model_name}.h5")
        print("Model saved successfully!")

        print(
            f'Saving history to "artifacts/model_histories/history_{self.model_name}"...'
        )
        with open(
            f"{ARTIFACT_DIR}/model_histories/history_{self.model_name}", "wb"
        ) as f:
            pickle.dump(history.history, f)
        print("History saved successfully!")

    def __load_history_or_model(
        self,
        path: str = "",
        history_or_model: Literal["history", "model"] = "history",
    ):
        """
        Private method to load either the model or its training history.

        Parameters
        ----------
        path : str, optional
            Path to the file, by default empty which will use the default path.
        history_or_model : Literal["history", "model"], optional
            Specify whether to load the "history" or the "model", by default "history".

        Raises
        ------
        Exception
            If the file is not found or if the file is not of the correct type.
        """

        if path == "":
            path = (
                f"{ARTIFACT_DIR}/model_histories/history_{self.model_name}"
                if history_or_model == "history"
                else f"{ARTIFACT_DIR}/models/{self.model_name}.h5"
            )

        print(f'Loading {history_or_model} from "{path}"...')

        model, history = None, None
        if history_or_model == "history":
            with open(path, "rb") as f:
                history = pickle.load(f)
        else:
            model = keras.models.load_model(path)

        if history_or_model == "history" and history is not None:
            self.history = history
        elif history_or_model == "model" and model is not None:
            self.model = model
        else:
            raise Exception(f"Error loading {history_or_model} from {path}!")

        print(f"{history_or_model.capitalize()} loaded successfully!")

    def load_model_and_history(self, model_path: str = "", history_path: str = ""):
        """
        Load the saved model and its training history.

        Parameters
        ----------
        model_path : str, optional
            Path to the saved model, by default empty.
        history_path : str, optional
            Path to the saved training history, by default empty.
        """

        self.__load_history_or_model(model_path, "model")
        self.__load_history_or_model(history_path, "history")

    def plot_accuracy(self):
        """
        Plot and save the training and validation accuracy of the model.

        Notes
        -----
        The plot is saved to "artifacts/plots/Accuracy_{self.model_name}.png".

        """
        plt.plot(self.history["accuracy"], label="train accuracy")
        plt.plot(self.history["val_accuracy"], label="validation accuracy")
        plt.title(f"{self.model_name} accuracy")
        plt.ylabel("Accuracy")
        plt.xlabel("Epoch")
        plt.legend(loc="best")
        plt.savefig(f"{ARTIFACT_DIR}/plots/Accuracy_{self.model_name}")
        plt.show()

    def plot_loss(self):
        """
        Plot and save the training and validation loss of the model.

        Notes
        -----
        The plot is saved to "artifacts/plots/Loss_{self.model_name}.png".
        """

        plt.plot(self.history["loss"], label="train loss")
        plt.plot(self.history["val_loss"], label="validation loss")
        plt.title(f"{self.model_name} loss")
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.legend(loc="best")
        plt.savefig(f"{ARTIFACT_DIR}/plots/Loss_{self.model_name}")
        plt.show()

    def __display_loss_and_accuracy(self):
        """
        Print out the model's loss and accuracy.
        """

        self.loss, self.accuracy = self.model.evaluate(
            self.preprocessing_utils.X_test, self.preprocessing_utils.y_test
        )

        print(f"Loss: {self.loss:.4f}")
        print(f"Accuracy: {self.accuracy:.4f}")

    def __display_confusion_matrix(self, y_pred: np.ndarray, y_true: np.ndarray):
        """
        Display the confusion matrix for the model's predictions.

        Parameters
        ----------
        y_pred : np.ndarray
            Array of predicted labels.
        y_true : np.ndarray
            Array of true labels.

        Notes
        -----
        The confusion matrix is saved to "artifacts/metrics/ConfusionMatrix_{self.model_name}.png".
        """

        cmap = "viridis"
        cm_plot_labels = [i for i in range(25)]

        cm = metrics.confusion_matrix(y_true=y_true, y_pred=y_pred)
        df_cm = pd.DataFrame(cm, cm_plot_labels, cm_plot_labels)
        sns.set(font_scale=1.1)
        plt.figure(figsize=(15, 10))
        s = sns.heatmap(df_cm, annot=True, cmap=cmap)
        plt.title(f"{self.model_name} confusion matrix")
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.savefig(f"{ARTIFACT_DIR}/metrics/ConfusionMatrix_{self.model_name}.png")
        plt.show()

    def display_evaluation_results(self):
        """
        Display the model's loss, accuracy, classification report, and confusion matrix.
        """

        self.__display_loss_and_accuracy()

        y_pred = self.model.predict(self.preprocessing_utils.X_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        y_true = np.argmax(self.preprocessing_utils.y_test, axis=1)

        print(
            f"Classification Report:\n{metrics.classification_report(y_true, y_pred_classes)}"
        )

        self.__display_confusion_matrix(y_pred_classes, y_true)

    def __predict_image_label(self, image_array: np.ndarray):
        """
        Private method to predict the label of an image.

        Parameters
        ----------
        image_array : np.ndarray
            The image converted to a numpy array to predict the label of.

        Returns
        -------
        tuple : (str, float)
            A tuple containing the predicted label and the confidence score.
        """

        img_processed = np.expand_dims(image_array.copy(), axis=0)

        prediction = self.model.predict(img_processed)

        index = np.argmax(prediction)
        confidence = prediction[0][index]

        prediction_label = self.preprocessing_utils.label_names[index]

        return prediction_label, confidence

    def display_random_image_prediction(self):
        """
        Visualize the model's prediction for a randomly chosen image from the test set.
        """

        idx = np.random.randint(len(self.preprocessing_utils.X_test))

        img_array = self.preprocessing_utils.X_test[idx]
        true_label = self.preprocessing_utils.label_names[
            np.argmax(self.preprocessing_utils.y_test[idx])
        ]

        prediction_label, confidence = self.__predict_image_label(img_array)

        plt.title(
            f"True label - {true_label}\nPredicted label - {prediction_label}\nConfidence - {confidence:.4f}"
        )
        plt.imshow(img_array)

    def display_all_image_predictions_for_label(self, label_name: str, limit: int = 5):
        """
        Visualize the model's predictions for all images of a specific label from the test set.

        Parameters
        ----------
        label_name : str
            The label name to filter the images by.
        limit : int, optional
            Size of the grid with dimensions (limit, limit), by default 5.
        """

        label_index = self.preprocessing_utils.label_names.index(label_name)

        one_hot_label = np.zeros_like(self.preprocessing_utils.y_test[0])
        one_hot_label[label_index] = 1

        filtered_images = [
            img
            for idx, img in enumerate(self.preprocessing_utils.X_test)
            if np.array_equal(self.preprocessing_utils.y_test[idx], one_hot_label)
        ]

        fig, axes = plt.subplots(limit, limit, figsize=(15, 15))

        for i, ax in enumerate(axes.ravel()):
            if i < len(filtered_images):
                prediction_label, confidence = self.__predict_image_label(
                    filtered_images[i]
                )

                ax.imshow(filtered_images[i])
                ax.set_title(
                    f"True: {label_name}\nPredicted: {prediction_label}\nConfidence: {confidence:.4f}",
                    fontsize=10,
                )
                ax.axis("off")

        plt.subplots_adjust(hspace=0.5)
        plt.show()
