from typing import Any, Tuple

import numpy as np
import torch


def inference_torch(
    model: Any[torch.nn.Module, torch.jit.ScriptModule],
    sample_input: Tuple[torch.Tensor],
) -> Tuple[np.ndarray]:
    """Inference Torch model.

    Args:
        model (Any[torch.nn.Module, torch.jit.ScriptModule]): Loaded Torch model instance.
        sample_input (Tuple[torch.Tensor]): Sample input to the model.

    Returns:
        Tuple[np.ndarray]: Model inference output
    """
    with torch.inference_mode():
        pred = model(sample_input)
    pred = pred.numpy()
    return pred
