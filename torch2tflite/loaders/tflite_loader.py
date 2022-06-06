import logging

import tensorflow as tf

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("tflite_loader")


def load_tflite(model_path: str) -> tf.lite.Interpreter:
    """Load TFLite model.

    Args:
        model_path (str): Path to the saved TFLite model.

    Returns:
        tf.lite.Interpreter: Loaded TFLite model.
    """
    interpret = tf.lite.Interpreter(model_path)
    interpret.allocate_tensors()
    logger.info(f"TFLite interpreter successfully loaded from, {model_path}")
    return interpret
