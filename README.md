# opt :dart:
This repository aims to provide implementations of modern optimization algorithms for machine learning in [TensorFlow 2.x](https://github.com/tensorflow/tensorflow), [Jax](https://jax.readthedocs.io/en/latest/) and [Trax](https://trax-ml.readthedocs.io/en/latest/), with the goal to reproduce experiments from the published research articles.


## Resume
**Momentum** : the files [opt/tensorflow/momentum.py](https://github.com/johanattia/opt/blob/master/opt/momentum.py), [opt/momentum.ipynb](https://github.com/johanattia/opt/blob/master/opt/momentum.ipynb) and [opt/tensorflow/schedule.py](https://github.com/johanattia/opt/blob/master/opt/schedule.py) allow to reproduce some experiments of the below articles. Particularly, this momentum optimizer implementation adds momentum schedule and thus extends native SGD of TensorFlow 2.x.
* [On the importance of initialization and momentum in deep learning](http://proceedings.mlr.press/v28/sutskever13.pdf), Ilya Sutskever, James Martens, George Dahl and Geoffrey Hinton.
* [Advances in Optimizing Recurrent Networks](https://arxiv.org/pdf/1212.0901.pdf), Yoshua Bengio, Nicolas Boulanger-Lewandowski and Razvan Pascanu.
