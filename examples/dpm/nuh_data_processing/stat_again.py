#!/usr/bin/python
#encode=utf8

patient = set()
time_l = []

f = open("shard_input", "rb")
to = open("stat_out", "w")

samples = f.readlines()
num_samples = len(samples)

with open("shard_input", "rb") as f:
    num = 0
    for item in f:
        time = 0
        f_and_l = item.split('\001')
        DELTA_T = f_and_l[1].split('\002')[1]
        #print "DELTA_T:%s\n" %DELTA_T
        time += int(DELTA_T)
        pid = f_and_l[1].split('\002')[0]
        patient.add(pid)
        for record in f_and_l[0].split('\002'):
            #print record
            delta_t = record.split('\003')[1]
            #print delta_t
            time += int(delta_t)
        time_l.append(time)
        #print time
        num += 1
        #if num > 1: break


num_patients = len(patient)
sum_t = 0.0
for t in time_l:
    sum_t += t
avg_time = sum_t / len(time_l)
to.write("Average time span: " + str(avg_time) + '\n')
to.write("Number of samples: " + str(num_samples) + '\n')
to.write("Number of patients: " + str(num_patients) + '\n')

f.close()
to.close()
