## PyTorch to TensorFlow Lite Converter

Converts PyTorch whole model into Tensorflow Lite

PyTorch -> Onnx -> Tensorflow 2 -> TFLite


Please install first


    python3 setup.py install
    
    
#### Args

- `--torch-path` Path to local PyTorch model, please save whole model e.g. [torch.save(model, PATH)](https://pytorch.org/tutorials/beginner/saving_loading_models.html#save-load-entire-model)
- `--tf-lite-path` Save path for Tensorflow Lite model
- `--target-shape` Model input shape to create static-graph (default: `(224, 224, 3`)
- `--sample-file` Path to sample image file. If model is not about computer-vision, please use leave empty and only
  enter `--target-shape`
- `--seed` Seeds RNG to produce random input data when `--sample-file` does not exists
- `--log=INFO` To see what happens behind

#### Basic usage of the script

To test with sample file:

    python3 -m torch2tflite.converter
        --torch-path tests/mobilenetv2_model.pt
        --tflite-path mobilenetv2.tflite
        --sample-file sample_image.png
        --target-shape 224 224 3

To test with random input to check gradients:

    python3 -m torch2tflite.converter
        --torch-path tests/mobilenetv2_model.pt
        --tflite-path mobilenetv2.tflite
        --target-shape 224 224 3
        --seed 10
