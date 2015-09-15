##Apache SINGA

Distributed deep learning system

[Project Website](http://singa.incubator.apache.org)

All the details can be found in project website.

This is the Model 1 - 1st version model:

Name: MultiMlpNaiiveCombine

Detailed info:

1. We use three time windows for features

2. Weight (importance) of each window is equal (all 0.333333...)

3. In the SoftmaxLossLayer, first compute the softmax results for each window and then obtain a weighted sum