# Coconet PyTorch Reimplementation
The aim of this project was to find alternative ways to generate music but I seem to have converged on the same solution as the original paper: [Counterpoint by Convolution](https://arxiv.org/abs/1903.07227) and trained with [Bach Chorales Dataset](https://arxiv.org/abs/1206.6392).  
You can see the original TensorFlow implementation in the [magenta GitHub repo](https://github.com/magenta/magenta/blob/main/magenta/models/coconet/).
## Install + Setup
This was developed and run using Python 3.10 and PyTorch 2.0.0 but might work with other versions.  
`pip install -r requirements.txt torch==2.0.0`  
To change the default settings for the script, modify the configuration at the top (global variables and configuration classes).
## Listen to Samples
Samples are available in `samples/`.
## Generate Samples
A pretrained model is available in `pretrained/`.  
`python generate.py`  
Samples are saved in `generated/`.
## Train a Model
`python train.py`  
Model iterations and training samples are saved in `checkpoints/`.
## Harmonise Melodies
Have a look in `harmonise.py`.  
Currently, the results are not great due to the *very* slow tempo of the Bach Chorales.  
The harmonised samples are saved in `harmonised/`.