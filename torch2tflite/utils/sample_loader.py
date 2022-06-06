from typing import Optional, Dict

import os
import sys
import logging

import numpy as np
import cv2
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sample_loader")


def load_sample_input(
    file_path: Optional[str] = None,
    target_shape: tuple = (224, 224, 3),
    seed: int = 10,
    normalize: bool = True,
) -> Dict[np.ndarray, torch.Tensor]:
    if os.path.exists(file_path) and os.path.isfile(file_path):
        if (len(target_shape) == 3 and target_shape[-1] == 1) or len(target_shape) == 2:
            imread_flags = cv2.IMREAD_GRAYSCALE
        elif len(target_shape) == 3 and target_shape[-1] == 3:
            imread_flags = cv2.IMREAD_COLOR
        else:
            imread_flags = cv2.IMREAD_ANYCOLOR + cv2.IMREAD_ANYDEPTH
        try:
            img = cv2.resize(
                src=cv2.imread(file_path, imread_flags),
                dsize=target_shape[:2],
                interpolation=cv2.INTER_LINEAR,
            )
            if len(img.shape) == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            if normalize:
                img = img * 1.0 / 255
            img = img.astype(np.float32)

            sample_data_np = np.transpose(img, (2, 0, 1))[np.newaxis, :, :, :]
            sample_data_torch = torch.from_numpy(sample_data_np)
            logger.info(f"Sample input successfully loaded from, {file_path}")

        except Exception:
            logger.error(f"Can not load sample input from, {file_path}")
            sys.exit(-1)

    else:
        logger.info("Sample input file path not specified, random data will be generated")
        np.random.seed(seed)
        data = np.random.random(target_shape).astype(np.float32)
        sample_data_np = np.transpose(data, (2, 0, 1))[np.newaxis, :, :, :]
        sample_data_torch = torch.from_numpy(sample_data_np)
        logger.info("Random sample input generated")

    return {
        "sample_data_np": sample_data_np,
        "sample_data_torch": sample_data_torch,
    }
