# This is  the code for printing confusion matrix For Classification

#!/usr/bin/python
import sys


if len(sys.argv) < 2:
    print "Pls input a file: python confus.py [your file]"
print "processing " + sys.argv[1]

c1 = [0,0,0]
c2 = [0,0,0]
c3 = [0,0,0]
accuracy = 0
num_samples = 0
with open(sys.argv[1], 'rb') as f:
    for line in f:
        num_samples += 1
        value = line.split(',')
        groudtruth = float(value[0].strip())
        prediction = float(value[1].strip())
        if groudtruth == 30:
            if prediction == 30:
                c1[0] += 1
                accuracy += 1
            elif prediction < 30 and prediction > 23:
                c1[1] += 1
            elif prediction <= 23:
                c1[2] += 1
        elif groudtruth < 30 and groudtruth > 23:
            if prediction == 30:
                c2[0] += 1
            elif prediction < 30 and prediction > 23:
                c2[1] += 1
                accuracy += 1
            elif prediction <= 23:
                c2[2] += 1
        elif groudtruth <= 23:
            if prediction == 30:
                c3[0] += 1
            elif prediction < 30 and prediction > 23:
                c3[1] += 1
            elif prediction <= 23:
                c3[2] += 1
                accuracy += 1

print "===================================="
print "Overall accurate number: %d"  % accuracy
print "Overall sample number: %d"  % num_samples
print 'c1: MMSE=30, c2: 23<MMSE<30, c3: MMSE<=23'
print "confusion matrix:"
print '\t' + 'c1' + '\t' + 'c2' + '\t' + 'c3'
print 'c1' + '\t' + str(c1[0]) + '\t' + str(c1[1]) + '\t' + str(c1[2])
print 'c2' + '\t' + str(c2[0]) + '\t' + str(c2[1]) + '\t' + str(c2[2])
print 'c3' + '\t' + str(c3[0]) + '\t' + str(c3[1]) + '\t' + str(c3[2])
print "===================================="
