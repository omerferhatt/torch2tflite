from setuptools import setup

# specify requirements of your package here
REQUIREMENTS = [
    'tensorflow ~= 2.5',
    'tflite-runtime ~= 2.5',
    'torch == 1.8.1',
    'tensorflow-addons ~= 0.13',
    'opencv-python ~= 4.5.2',
    'onnx ~= 1.9',
    'onnx-tf ~= 1.8',
    'numpy >= 1.19'
]

# some more details
CLASSIFIERS = [
    'Development Status :: 4 - Beta',
    'Intended Audience :: Developers',
    'Topic :: Machine Learning',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8',
]

# calling the setup function
setup(name='torch2tflite',
      version='1.0.0',
      description='PyTorch to TFLite model converter',
      url='https://github.com/omerferhatt/torch2tflite',
      author='Omer F. Sarioglu',
      author_email='omerf.sarioglu@gmail.com',
      license='MIT',
      packages=['torch2tflite'],
      classifiers=CLASSIFIERS,
      install_requires=REQUIREMENTS,
      keywords='torch pytorch tensorflow tflite converter onnx machine-learning'
      )
