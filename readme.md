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
where the suffix used as output from the autopicker was _autopick.star, the files with correct picks have the suffix _autopick_correct.star and the output suffix you want is _autopick_train.star. This script iterates through all files in the currect directory that have the _autopick_correct.star suffix, generating a _autopick_train.star file which contains an extra column that labels whether the coordinates of the box represent a particle ('P' - for all particles in _autopick_correct.star) or something else ('N' - for coordinates only found in the _autopick.star). 

...