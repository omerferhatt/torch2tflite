import argparse
from converter.torch_to_tflite import *


def init_models(torch_model_path, tf_lite_model_path):
    """
    Initialize the Torch and TFLite models
    :param torch_model_path: Path to Torch model
    :param tf_lite_model_path: Path to TFLite model
    :return: CPU initialized models
    """
    torch_model = get_torch_model(torch_model_path)
    tf_lite_model = get_tf_lite_model(tf_lite_model_path)
    return torch_model, tf_lite_model


def main():
    """
    Converts PyTorch model into TFLite with using ONNX and main TensorFlow between them.
    """
    if args.convert:
        convert(torch_model_path=args.torch_model_path,
                tf_lite_model_path=args.tf_lite_model_path,
                image_path=args.test_im_path)

    if args.show_results or not args.convert:
        print('Showing result!\n')
        # Loads image according to model input types
        original_image, tf_lite_image, torch_image = get_example_input(args.test_im_path)
        # Initialize the both model
        torch_model, tf_lite_model = init_models(args.torch_model_path, args.tf_lite_model_path)
        # Inference with models
        tf_lite_output = predict_tf_lite(tf_lite_model, tf_lite_image)
        torch_output = predict_torch(torch_model, torch_image)
        # Calculates loss metrics of outputs between two model
        _ = calc_error(tf_lite_output, torch_output, verbose=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch->TFLite Converter')

    # Paths
    parser.add_argument('--torch-model-path', type=str, default='./torch_model/model.pt',
                        help='Path to the torch model (*.pt) file.')
    parser.add_argument('--tf-lite-model-path', type=str, default='./converter/tf_lite_model.tflite',
                        help='Path to the TFLite model (*.tflite) file.')
    parser.add_argument('--test-im-path', type=str, default='./converter/test_images/test.png',
                        help='Path to test image.')
    # Results
    parser.add_argument('--show-results', action='store_true', default=False,
                        help='Shows comparison and metrics between two models')
    # Convert
    parser.add_argument('--convert', action='store_true', default=False,
                        help='Converts Torch model into TensorFlow Lite')

    args = parser.parse_args()

    main()
