from loguru import logger
from flask import Flask, request, jsonify

import lib as model_lib

app = Flask(__name__)


# Initialize model library
_MODEL_PATH: str = "model.pkl"
try:
    model_lib.init_lib(_MODEL_PATH)
    logger.info("Model initialized successfully.")
except Exception as e:
    logger.error(f"Failed to initialize model: {e}")
    raise


def is_message_empty(message: str) -> bool:
    """
    Check if the message is empty or contains only whitespace.

    Args:
        message (str): The message to check.

    Returns:
        bool: True if the message is empty or contains only whitespace, False otherwise.
    """
    return not message.strip()


def contains_alpha(message: str) -> bool:
    """
    Check if the message contains at least one alphabetic character.

    Args:
        message (str): The message to check.

    Returns:
        bool: True if the message contains at least one alphabetic character, False otherwise.
    """
    return any(char.isalpha() for char in message)


@app.route("/predict", methods=["PUT"])
def predict_put():
    """
    Predict the probability of SPAM for an SMS message.

    Returns:
        JSON response with SPAM probability or error message.
    """
    if not request.is_json:
        return jsonify({"error": "Invalid input format. JSON expected."}), 400

    try:
        body = request.get_json()

        # Validate JSON structure
        if not isinstance(body, dict):
            return jsonify({"error": "Invalid input format. JSON body must be a dictionary."}), 400

        message = body.get("message", "")
        if not isinstance(message, str):
            return jsonify({"error": "Invalid input format. 'message' must be a string."}), 400

        # Validate message content
        if is_message_empty(message):
            return jsonify({"error": "Empty input - no alphanumeric characters"}), 400
        if not contains_alpha(message):
            return jsonify({"error": "Malformed input - no alphabetic characters"}), 400

        # Predict the probability
        try:
            y_pred_prob = model_lib.predict_probability(message)
        except Exception as predict_err:
            logger.error(f"Error predicting probability: {predict_err}")
            return jsonify({"error": f"Error predicting probability: {predict_err}"}), 500

        # Return the prediction result
        return jsonify({"SPAM_probability": y_pred_prob})

    except Exception as err:
        logger.error(f"Internal server error: {err}")
        return jsonify({"error": f"Internal server error: {err}"}), 500


if __name__ == "__main__":
    app.run(debug=True)
