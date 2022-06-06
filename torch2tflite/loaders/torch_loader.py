import logging

import torch
import sys

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("torch_loader")


def torch_loader(model_path: str) -> torch.nn.Module:
    """Load default saved PyTorch model.

    Please be sure torch.save() was used to save the model.

    Args:
        model_path (str): Path to the saved PyTorch model.

    Returns:
        torch.nn.Module: Loaded PyTorch model.
    """
    try:
        model = torch.load(model_path, map_location="cpu")
    except Exception:
        logger.error("Can not load PyTorch model. Please make sure that model saved successfully.")
        sys.exit(-1)
    model = model.eval()
    logger.info("PyTorch model loaded and mapped to CPU successfully.")
    return model


def torchscript_loader(model_path: str) -> torch.jit.ScriptModule:
    """Load default saved TorchScript model.

    Please be sure torch.jit.save() was used to save the model.

    Args:
        model_path (str): Path to the saved TorchScript model.

    Returns:
        torch.jit.ScriptModule: Loaded TorchScript model.
    """
    try:
        model = torch.jit.load(model_path, map_location="cpu")
    except Exception:
        logger.error("Can not load TorchScript model. Please make sure that model saved successfully.")
        sys.exit(-1)
    model = model.eval()
    logger.info("TorchScript model loaded and mapped to CPU successfully.")
    return model
