# Jacobian Regularisation for Convolutional Neural Networks 

## A Text Classification Example

### About Code
This is currently a work in progress. See notebooks for examples and test runs. Evaluation results coming soon.

### How to run
- Download the [IMdB Movie Reviews](http://ai.stanford.edu/~amaas/data/sentiment/) and [GloVe](https://nlp.stanford.edu/projects/glove/) datasets.
- Generate embeddings using: 

`python embeddings.py -d data/glove.42B.300d.txt --npy_output data/embeddings.npy --dict_output data/vocab.pckl --dict_whitelist data/aclImdb/imdb.vocab`

- Start training with train.py

### About Model

A Deep Convolutional Neural Network architecture, with a custom regularisation function to bound the norm of network’s Jacobian in the neighbourhood of training samples for tighter generalisation [1].

### References

[1]  [Robust Large Margin Deep Neural Networks](https://arxiv.org/abs/1605.08254), Jure Sokolic et al. (revised May 2017)


### Sources

[1]  Jacobian Regularizer: [jureso/RobustLargeMarginDNN](https://github.com/jureso/RobustLargeMarginDNN)

[2]  TextCNN: [dennybritz/cnn-text-classification-tf](https://github.com/dennybritz/cnn-text-classification-tf)

[3]  Data Helpers: [rampage644/qrnn](https://github.com/rampage644/qrnn)

