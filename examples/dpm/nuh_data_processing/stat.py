#!/usr/bin/python
#encode=utf-8
import re

shard = open("shard_input", "rb")

avg = 0
var = 0
sum = 0
sq_sum = 0
num = 0

list = []
for item in shard.readlines():
    curr = item.split('\001')
    curr2 = curr[0].split('\003')
    record = int(curr2[-1])
    list.append(record)
    sum += record
    sq_sum += record * record
    num += 1

avg = float(sum) / num
print "average is: " + str(avg)
print "variance is: " + str(sq_sum - num * avg * avg)

sorted(list)

cnt = 0
cnt1 = 0
cnt2 = 0
cnt3 = 0
cnt4 = 0
cnt5 = 0
cnt6 = 0
cnt7 = 0
cnt8 = 0
for item in list:
    if item > 500:
        cnt += 1
    elif item > 300:
        cnt1 += 1
    elif item > 200:
        cnt2 += 1
    elif item > 100:
        cnt3 += 1
    elif item > 50:
        cnt4 += 1
    elif item > 30:
        cnt5 += 1
    elif item > 20:
        cnt6 += 1
    elif item > 10:
        cnt7 += 1
    else:
        cnt8 += 1

print "> 500: %d" %cnt
print "> 300: %d" %cnt1
print "> 200: %d" %cnt2
print "> 100: %d" %cnt3
print "> 50: %d" %cnt4
print "> 30: %d" %cnt5
print "> 20: %d" %cnt6
print "> 10: %d" %cnt7
print "< 10: %d" %cnt8
