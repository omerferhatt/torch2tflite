import onnx
import onnx_tf


def onnx2tf(model_path: str, save_path: str) -> None:
    """Convert ONNX model to TensorFlow model.

    Args:
        model_path (str): Path to ONNX model.
        save_path (str): Path to save TensorFlow model.
    
    Returns:
        None
    """
    onnx_model = onnx.load(model_path)
    onnx.checker.check_model(onnx_model)
    tf_rep = onnx_tf.backend.prepare(onnx_model)
    tf_rep.export_graph(save_path)
