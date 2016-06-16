#emUnpicker - tools to automate choosing correct particles in electron micrographs after autopicking
These scripts automate unpicking incorrect picks in electron micrographs after autopicking. They do this by using the [TensorFlow](https://www.tensorflow.org/) toolkit to train a neural network based on a set of correct picks. These scripts are intended to complement the [RELION](http://www2.mrc-lmb.cam.ac.uk/relion/index.php/Main_Page) autopicker as they use .star files to describe the box coordinates, but should work with any autopicker whose output can be converted to this format.

##Installation
###Prerequisites
- [TensorFlow](https://www.tensorflow.org/)
- [SciPy](https://www.scipy.org/scipylib/index.html)
- [PIL](http://www.pythonware.com/products/pil/)

Just clone this repository and check that the script trainingSetFromStarFiles.py is exectutable.

##Usage
The scripts can be run with or without ctf correction. In practice, operating without CTF correction is just as accurate and runs quicker.
###No CTF correction
Good results have been obtained having between 1,000 - 10,000 correct particles as input for training. To obtain these particles, copy the _autopick.star (e.g. to _autopick_correct.star) and unpick until you have between 1,000 - 10,000 particles. Then run `trainingSetFromStarFiles.py` from within your micrographs directory. Running this without arguments shows the input that this script expects. As an example:
```
trainingSetFromStarFiles.py _autopick.star _autopick_correct.star _autopick_train.star
``` 
where the suffix after the autopicker was _autopick.star, the files with correct picks have the suffix _autopick_correct.star and the output suffix you want is _autopick_train.star. This script iterates through all files in the currect directory that have the _autopick_correct.star suffix, generating a _autopick_train.star file which contains an extra column that labels whether the coordinates of the box represent a particle ('P' - for all particles in _autopick_correct.star) or something else ('N' - for coordinates only found in the _autopick.star). 

It is important that the training set is representative of your images as a whole. The images chosen to manually unpick should cover the range of defocus values, any differences in ice thickness and distribution of your particles. If there is a bias, the unpicker will work really well for some of your images and poorly for the rest.

The next step is to train the network. The best option is to use the unpicker_train.py script.
```
$ python /em/Scripts/emUnpicker/unpicker_train.py --help
usage: unpicker_train.py [-h] [--train_root TRAIN_ROOT]
                         [--train_output TRAIN_OUTPUT] [--boxsize BOXSIZE]
                         [--sigma_contrast SIGMA_CONTRAST] [--apix APIX]
                         [--lowpass LOWPASS] [--highpass HIGHPASS]
                         [--gaussian_sigma GAUSSIAN_SIGMA]
                         [--num_cores NUM_CORES] [--num_epochs NUM_EPOCHS]
                         [--resized_box RESIZED_BOX]

optional arguments:
  -h, --help            show this help message and exit
  --train_root TRAIN_ROOT
                        File suffix for star files with particle training data
  --train_output TRAIN_OUTPUT
                        File to save training data
  --boxsize BOXSIZE     Boxsize to use when extracting particles
  --sigma_contrast SIGMA_CONTRAST
                        Sigma contrast to apply to image before particle
                        extraction
  --apix APIX           Angstroms per pixel in mrc file
  --lowpass LOWPASS     Lowpass filter resolution to apply
  --highpass HIGHPASS   Highpass filter resolution to apply
  --gaussian_sigma GAUSSIAN_SIGMA
                        Gaussian filter sigma to apply
  --num_cores NUM_CORES
                        Number of cores to use
  --num_epochs NUM_EPOCHS
                        Number of epochs for training
  --resized_box RESIZED_BOX
                        Resize boxsize used for training
```
This script should also be run in the Micrographs directory and initially loads all the training data - both particles and non-particles. These are then processed according to the options given. The sigma_contrast variable (which is identical to sigma contrast within RELION) has a large effect on the accuracy of the training and it is wise to screen a variety of values. Other image processing options don't seem to improve the training accuracy, but will be dependent on the data. Tensorflow can be run on a CPU multi-threaded (controlled with --num_cores) or use a gpu if present (giving a large speed increase for training).

Example run:
```
$ python /em/Scripts/emUnpicker/unpicker_train.py --train_root _autopick_train.star --train_output unpickModel.out --boxsize 250 --sigma_contrast 5 --num_cores 6 --num_epochs 25
Training Particles: 4239 Non-particles: 4239 in training set
Loaded training data, using 3391 particles for validation - 50.0% particles
I tensorflow/core/common_runtime/local_device.cc:25] Local device intra op parallelism threads: 6
I tensorflow/core/common_runtime/local_session.cc:45] Local session inter op parallelism threads: 6
Initialized!
Epoch 0.00
Step 0
Minibatch loss: 4.963, learning rate: 0.010000
Minibatch error: 46.9%
Validation error: 54.4%
Step 0 model saved in file unpickModel.out-0
Epoch 0.21
Step 100
Minibatch loss: 4.585, learning rate: 0.010000
Minibatch error: 32.8%
Validation error: 30.7%
Step 100 model saved in file unpickModel.out-100
Epoch 0.42
Step 200
Minibatch loss: 4.612, learning rate: 0.010000
Minibatch error: 37.5%
Validation error: 30.3%
Step 200 model saved in file unpickModel.out-200
...
Epoch 24.53
Step 11700
Minibatch loss: 2.347, learning rate: 0.002920
Minibatch error: 12.5%
Validation error: 17.0%
Step 11700 model saved in file unpickModel.out-11700
Epoch 24.74
Step 11800
Minibatch loss: 2.319, learning rate: 0.002920
Minibatch error: 9.4%
Validation error: 17.6%
Step 11800 model saved in file unpickModel.out-11800
Epoch 24.95
Step 11900
Minibatch loss: 2.283, learning rate: 0.002920
Minibatch error: 10.9%
Validation error: 18.0%
Step 11900 model saved in file unpickModel.out-11900
Training completed, final validation error: 19.7%
Trained model saved in file unpickModel.out
```
Initially the script loads all the training data, applies the desired processing and generates four 90 degree rotations of each box. From this a subset of the training data is extracted that is just used for validation - the network is not trained using this data and serves to provide validation of the training. As the cycles progress, it is expected that the validation error will reduce. Multiple training runs will be required to determine the optimal values. The final network weight information is saved in the train_output file - in this case unpickModel.out. The script also saves a checkpoint file every 100 steps and any of these output files (i.e. the one with the lowest validation error) can be used for evaluation.

Once the network weights with the lowest validation error have been obtained, all the images in the dataset are unpicked using unpicker_eval.py. This script has similar options to the training script:
```
$ python /em/Scripts/emUnpicker/unpicker_eval.py --help
usage: unpicker_eval.py [-h] [--train_output TRAIN_OUTPUT]
                        [--eval_file EVAL_FILE] [--eval_root EVAL_ROOT]
                        [--output_root OUTPUT_ROOT] [--boxsize BOXSIZE]
                        [--sigma_contrast SIGMA_CONTRAST] [--apix APIX]
                        [--lowpass LOWPASS] [--highpass HIGHPASS]
                        [--gaussian_sigma GAUSSIAN_SIGMA]
                        [--num_cores NUM_CORES] [--resized_box RESIZED_BOX]

optional arguments:
  -h, --help            show this help message and exit
  --train_output TRAIN_OUTPUT
                        File with training data
  --eval_root EVAL_ROOT
                        File suffix for star files to unpick
  --output_root OUTPUT_ROOT
                        File suffix for unpicked star files
  --boxsize BOXSIZE     Boxsize to use when extracting particles
  --sigma_contrast SIGMA_CONTRAST
                        Sigma contrast to apply to image before particle
                        extraction
  --apix APIX           Angstroms per pixel in mrc file
  --lowpass LOWPASS     Lowpass filter resolution to apply
  --highpass HIGHPASS   Highpass filter resolution to apply
  --gaussian_sigma GAUSSIAN_SIGMA
                        Gaussian filter sigma to apply
  --num_cores NUM_CORES
                        Number of cores to use
  --resized_box RESIZED_BOX
                        Resize boxsize used for training
```
To unpick the dataset, run the unpicker_eval.py script with the output from training and the same image processing parameters. Changing any image processing parameters will invalidate the training data!

...