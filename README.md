# Jacobian Regularisation for Convolutional Neural Networks 

## A Text Classification Example with Character Level ConvNets

### About Code
This is currently a work in progress. See notebooks for examples and test runs. Evaluation results coming soon.

### Downloads
- [Yelp Reviews](https://www.yelp.com/dataset/challenge)

### About Model

A character level Convolutional Neural Network architecture<sup>[1]</sup>, with a custom regularisation function to bound the norm of network’s Jacobian in the neighbourhood of training samples for tighter generalisation<sup>[2]</sup>.

### References

[1]  [Character-level Convolutional Networks for Text Classification](https://papers.nips.cc/paper/5782-character-level-convolutional-networks-for-text-classification.pdf), Zhang et al. (September 2015)

[2]  [Robust Large Margin Deep Neural Networks](https://arxiv.org/abs/1605.08254), Jure Sokolic et al. (revised May 2017)


### Sources

- Jacobian Regularizer: [jureso/RobustLargeMarginDNN](https://github.com/jureso/RobustLargeMarginDNN)

- CharCNN: [scharmchi/char-level-cnn-tf](https://github.com/scharmchi/char-level-cnn-tf)

