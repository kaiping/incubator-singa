##Apache SINGA

Distributed deep learning system

[Project Website](http://singa.incubator.apache.org)

All the details can be found in project website.

This is the Model 1 - 2nd version model:

Name: MultiMlpNaiiveCombine-Var1

Detailed info:

1. We use three time windows for features

2. Weight (importance) of each window is not equal (the weight for win1 is 1.0/6, for win2 is 1.0/3, and for win3 is 1.0/2)

The basic idea for this is to give higher weight to the more recent window;

The only difference between this version and Model1-1st version lies in 
that the computation (both forward and backward) for the DPMCombineSoftmaxLossLayer

3. In the SoftmaxLossLayer, first compute the softmax results for each window and then obtain a weighted sum