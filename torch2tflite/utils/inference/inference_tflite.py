from typing import Tuple

import numpy as np
import tensorflow as tf


def inference_tflite(
    model: tf.lite.Interpreter,
    sample_input: Tuple[np.ndarray],
) -> Tuple[np.ndarray]:
    """Inference TFLite model.

    Args:
        model (tf.lite.Interpreter): Loaded TFLite model instance.
        sample_input (Any[Tuple[np.ndarray], np.ndarray]): Sample input to the model.

    Returns:
        np.ndarray: Model inference output
    """
    input_details = model.get_input_details()
    output_details = model.get_output_details()
    for inp_idx, inp in enumerate(input_details):
        model.set_tensor(inp["index"], sample_input[inp_idx])
    model.invoke()

    pred = []
    for out in output_details:
        pred.append(model.get_tensor(out["index"]))

    return tuple(pred)
