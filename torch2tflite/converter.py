from typing import Optional

import os
import sys
import shutil
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("torch2tflite")


class Torch2TFLiteConverter:
    def __init__(
        self,
        torch_model_path: str,
        tflite_model_save_path: str,
        sample_file_path: Optional[str] = None,
        target_shape: tuple = (224, 224, 3),
        seed: int = 10,
        normalize: bool = True,
    ):
        self.torch_model_path = torch_model_path
        self.tflite_model_path = tflite_model_save_path
        self.sample_file_path = sample_file_path
        self.target_shape = target_shape
        self.seed = seed
        self.normalize = normalize

        self.tmpdir = "/tmp/torch2tflite/"
        self.__check_tmpdir()
        self.onnx_model_path = os.path.join(self.tmpdir, "model.onnx")
        self.tf_model_path = os.path.join(self.tmpdir, "tf_model")
        self.torch_model = self.load_torch_model()
        self.sample_data = self.load_sample_input(sample_file_path, target_shape, seed, normalize)

    def convert(self):
        self.torch2onnx()
        self.onnx2tf()
        self.tf2tflite()
        torch_output = self.inference_torch()
        tflite_output = self.inference_tflite(self.load_tflite())
        self.calc_error(torch_output, tflite_output)

    def __check_tmpdir(self):
        try:
            if os.path.exists(self.tmpdir) and os.path.isdir(self.tmpdir):
                shutil.rmtree(self.tmpdir)
                logger.info("Old temp directory removed")
            os.makedirs(self.tmpdir, exist_ok=True)
            logger.info(f"Temp directory created at {self.tmpdir}")
        except Exception:
            logger.error("Can not create temporary directory, exiting!")
            sys.exit(-1)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--torch-path", type=str, required=True)
    parser.add_argument("--tflite-path", type=str, required=True)
    parser.add_argument("--target-shape", type=tuple, nargs=3, default=(224, 224, 3))
    parser.add_argument("--sample-file", type=str)
    parser.add_argument("--seed", type=int, default=10)
    args = parser.parse_args()

    conv = Torch2TFLiteConverter(
        args.torch_path,
        args.tflite_path,
        args.sample_file,
        args.target_shape,
        args.seed,
    )
    conv.convert()
    sys.exit(0)
