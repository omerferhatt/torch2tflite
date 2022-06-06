import tensorflow as tf


def tf2tflite(model_path: str, save_path: str) -> None:
    """Convert TensorFlow model to TensorFlow Lite model.

    Args:
        model_path (str): Path to TensorFlow model.
        save_path (str): Path to save TensorFlow Lite model.

    Returns:
        None
    """
    converter = tf.lite.TFLiteConverter.from_saved_model(model_path)
    tflite_model = converter.convert()
    with open(save_path, "wb") as f:
        f.write(tflite_model)
