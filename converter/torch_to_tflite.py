import os
import shutil
import sys

import cv2
import numpy as np
import onnx
import tensorflow as tf
import torch
from PIL import Image
from onnx_tf.backend import prepare
from torchvision import transforms


# ------------------ Image IO ------------------ #
def get_example_input(image_file):
    """
    Loads image from disk and converts to compatible shape.
    :param image_file: Path to single image file
    :return: Original image, numpy.ndarray instance image, torch.Tensor image
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    image = cv2.imread(image_file)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(image)
    torch_img = transform(pil_img)
    torch_img = torch_img.unsqueeze(0)
    torch_img = torch_img.to(torch.device("cpu"))

    return image, torch_img.numpy(), torch_img


# ------------------ Convert Functions ------------------ #
def torch_to_onnx(torch_path, onnx_path, image_path):
    """
    Converts PyTorch model file to ONNX with usable op-set
    :param torch_path: Torch model path to load
    :param onnx_path: ONNX model path to save
    :param image_path: Path of test image to use in export progress
    """
    pytorch_model = get_torch_model(torch_path)
    image, tf_lite_image, torch_image = get_example_input(image_path)

    torch.onnx.export(
        model=pytorch_model,
        args=torch_image,
        f=onnx_path,
        verbose=False,
        export_params=True,
        do_constant_folding=False,  # fold constant values for optimization
        input_names=['input'],
        opset_version=10,
        output_names=['output'])


def onnx_to_tf(onnx_path, tf_path):
    """
    Converts ONNX model to TF 2.X saved file
    :param onnx_path: ONNX model path to load
    :param tf_path: TF path to save
    """
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)  # Checks signature
    tf_rep = prepare(onnx_model)  # Prepare TF representation
    tf_rep.export_graph(tf_path)  # Export the model


def tf_to_tf_lite(tf_path, tf_lite_path):
    """
    Converts TF saved model into TFLite model
    :param tf_path: TF saved model path to load
    :param tf_lite_path: TFLite model path to save
    """
    converter = tf.lite.TFLiteConverter.from_saved_model(tf_path)  # Path to the SavedModel directory
    tflite_model = converter.convert()  # Creates converter instance
    with open(tf_lite_path, 'wb') as f:
        f.write(tflite_model)


# ------------------ Model Load Functions ------------------ #
def get_torch_model(model_path):
    """
    Loads state-dict into model and creates an instance
    :param model_path: State-dict path to load PyTorch model with pre-trained weights
    :return: PyTorch model instance
    """
    model = torch.load(model_path, map_location='cpu')
    return model


def get_tf_lite_model(model_path):
    """
    Creates an instance of TFLite CPU interpreter
    :param model_path: TFLite model path to initialize
    :return: TFLite interpreter
    """
    interpret = tf.lite.Interpreter(model_path)
    interpret.allocate_tensors()
    return interpret


# ------------------ Inference Functions ------------------ #
def predict_torch(model, image):
    """
    Torch model prediction (forward propagate)
    :param model: PyTorch model
    :param image: Input image
    :return: Numpy array with logits
    """
    return model(image).data.cpu().numpy()


def predict_tf_lite(model, image):
    """
    TFLite model prediction (forward propagate)
    :param model: TFLite interpreter
    :param image: Input image
    :return: Numpy array with logits
    """
    input_details = model.get_input_details()
    output_details = model.get_output_details()
    model.set_tensor(input_details[0]['index'], image)
    model.invoke()
    tf_lite_output = model.get_tensor(output_details[0]['index'])
    return tf_lite_output


def calc_error(res1, res2, verbose=False):
    """
    Calculates specified error between two results. In here Mean-Square-Error and Mean-Absolute-Error calculated"
    :param res1: First result
    :param res2: Second result
    :param verbose: Print loss results
    :return: Loss metrics as a dictionary
    """
    mse = ((res1 - res2) ** 2).mean(axis=None)
    mae = np.abs(res1 - res2).mean(axis=None)
    metrics = {'mse': mse, 'mae': mae}
    if verbose:
        print(f"\n\nMean-Square-Error between predictions:\t{metrics['mse']}")
        print(f"Mean-Square-Error between predictions:\t{metrics['mae']}\n\n")
    return metrics


# ------------------ Main Convert Function ------------------#
def convert(torch_model_path, tf_lite_model_path, image_path):
    if os.path.exists('output'):
        shutil.rmtree('output')
        os.mkdir('output')
    else:
        os.mkdir('output')
    ONNX_PATH = "output/onnx_model.onnx"
    TF_PATH = "output/tf_model"

    try:
        torch_to_onnx(torch_path=torch_model_path, onnx_path=ONNX_PATH, image_path=image_path)
        print('\n\nTorch to ONNX converted!\n\n')
    except Exception as e:
        print(e)
        sys.exit(1)
    try:
        onnx_to_tf(onnx_path=ONNX_PATH, tf_path=TF_PATH)
        print('\n\nONNX to TF converted!\n\n')
    except Exception as e:
        print(e)
        sys.exit(1)
    try:
        tf_to_tf_lite(tf_path=TF_PATH, tf_lite_path=tf_lite_model_path)
        print('\n\nTF to TFLite converted!\n\n')
    except Exception as e:
        print(e)
        sys.exit(1)
