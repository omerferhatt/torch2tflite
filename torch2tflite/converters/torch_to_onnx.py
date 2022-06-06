from typing import Any, Tuple, List

import torch


def torch2onnx(
    model_path: str,
    save_path: str,
    sample_input: Any[Tuple[torch.Tensor], torch.Tensor],
    input_names: Any[str, List[str]],
    output_names: Any[str, List[str]],
    opset_version: int,
    verbose: bool = False,
    export_params: bool = True,
    do_constant_folding: bool = False,
) -> None:
    """Convert PyTorch model to ONNX model.

    Args:
        model_path: Path to PyTorch model.
        save_path: Path to save ONNX model.
        sample_input: Sample input to the model.
        input_names: Input names of the model.
        output_names: Output names of the model.
        opset_version: Opset version for ONNX converter.
        verbose: Verbose for exporter. Defaults to False.
        export_params: Export additional params with exporter. Defaults to True.
        do_constant_folding: Do constant folding while exporting. Defaults to False.

    Returns:
        None
    """
    torch.onnx.export(
        model=model_path,
        f=save_path,
        args=sample_input,
        input_names=input_names,
        output_names=output_names,
        opset_version=opset_version,
        verbose=verbose,
        export_params=export_params,
        do_constant_folding=do_constant_folding,
    )
