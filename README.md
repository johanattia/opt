# opt :dart:

This repository aims to provide implementations of modern optimization algorithms for machine learning in [TensorFlow 2.x](https://github.com/tensorflow/tensorflow), with the goal to reproduce experiments from the published research articles.


## Optimizers

### Momentum

This momentum optimizer implementation adds momentum schedule and thus extends built-in SGD of TensorFlow 2.x. It is based on the following articles :
* [On the importance of initialization and momentum in deep learning](http://proceedings.mlr.press/v28/sutskever13.pdf), Ilya Sutskever, James Martens, George Dahl and Geoffrey Hinton.
* [Advances in Optimizing Recurrent Networks](https://arxiv.org/pdf/1212.0901.pdf), Yoshua Bengio, Nicolas Boulanger-Lewandowski and Razvan Pascanu.

See [examples](https://github.com/johanattia/opt/blob/master/opt/examples) folder for reproduced experiments.
