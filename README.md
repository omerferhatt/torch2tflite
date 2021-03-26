## PyTorch to TensorFlow Lite Converter

It uses ONNX and TF2 as bridge between Torch and TFLite

Torch -> ONNX -> TF2 -> TFLite

#### Args

- `--torch-model-path`: Path to local PyTorch model, e.g. `torch_models/torch_model.pt` (default)
- `--tf-lite-model-path`: Path to TFLite model, it can be usable as a save path and load path same time,
  e.g. `output/tf_lite_model.tflite` (default)
- `--test-im-path`: Single image path to test converted model on it, e.g. `sample/test.png` (default)
- `--show-results`: Shows error between models, e.g. `False` (default)
- `--convert`: Converts model, if this argument is false. It's only shows comparison between old converted models.
  e.g. `True`

#### Basic usage of the script

Convert models and show results:

    python3 converter.py \
    --torch-model-path torch_models/torch_model.pt \
    --tf-lite-model-path output/tf_lite_model.tflite \
    --test-im-path sample/test.png \
    --show-results \
    --convert
    

#### Required libraries

    tensorflow>=2.3.1 (conda)
    tensorflow-addons>=0.11.2 (pip)
    pytorch>=1.7.0 (cpu-only) (conda channel=pytorch)
    onnx>=1.8.0 (pip)
    onnx-tf>=1.7.0 (pip channel=git repo)
