import pickle
from typing import Any

_MODEL: Any = None


def _load_model(model_path: str) -> Any:
    """
    Load a model from a given path using pickle.

    Args:
        model_path (str): Path to the model file.

    Returns:
        Any: The loaded model.
    """
    try:
        with open(model_path, "rb") as input_stream:
            model = pickle.load(input_stream)
    except FileNotFoundError:
        raise FileNotFoundError(f"Model file not found at path: {model_path}")
    except pickle.UnpicklingError:
        raise ValueError("Error unpickling the model. The file might be corrupted or not a valid pickle file.")
    except Exception as e:
        raise Exception(f"An error occurred while loading the model: {e}")

    return model


def init_lib(model_path: str) -> None:
    """
    Initialize the library by loading the model from the given path.

    Args:
        model_path (str): Path to the model file.
    """
    global _MODEL
    _MODEL = _load_model(model_path)


def predict_probability(message: str) -> float:
    """
    Predict the probability of a message using the loaded model.

    Args:
        message (str): The input message to predict.

    Returns:
        float: The predicted probability.

    Raises:
        ValueError: If the model has not been initialized.
    """
    if _MODEL is None:
        raise ValueError("The model is not initialized. Call init_lib(model_path) first.")

    try:
        return float(_MODEL.predict_proba([message])[:, 1])
    except AttributeError:
        raise AttributeError("The model does not support the predict_proba method.")
    except Exception as e:
        raise Exception(f"An error occurred during prediction: {e}")
