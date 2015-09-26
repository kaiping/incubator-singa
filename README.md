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

Note:

The example under /examples/dpm_model1 is modified as follows:
Add a output_info.cc file to output all features as well as the label for the patient for WEKA experiments
Modify the Makefile to compile and run output_info.cc
Next to create shard, need to 
--cp from Makefile.example to Makefile--