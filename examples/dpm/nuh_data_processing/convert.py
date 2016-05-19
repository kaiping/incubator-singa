#!/usr/bin/env python
from datetime import datetime
import sys, time
import collections
#import csv

CtrlA = '\001'
CtrlB = '\002'
CtrlC = '\003'
CtrlD = '\004'
threshold = 50 # max num of record for data shard
#DAY = 24 * 3600 # seconds of a day

Features = "./features"
Labels = "./labels"
Demographics = "./demographics"
Output = './shard_input'

shard = open(Output, 'w')
#posix_now = time.time()

#d = datetime.fromtimestamp(posix_now)
vocab = {}

#colname = ('code','dtime','symbol','labtest')
colname = ('code', 'value', 'time')
feature = {}
demo = {}
codeid = {} # feature encode
patientid = {} # patient id encode

def read_features():
  '''
  fill up feature, codeid and patientid, from Features
  '''
  with open(Features, 'r') as f:

    numline = 0
    codenum = 0
    patientnum = 0

    for line in f:
      words = line[:-1].split(',')
      plist = []
      for i in range(1, len(words)):
        p = {}
        if not ')' in words[i]:
          continue
        else:
          words[i] = words[i].replace(')', '').replace('\'', '')
          p[colname[2]] = datetime.strptime(words[i].strip(), '%Y-%m-%d %H:%M:%S') #time
          p[colname[1]] = words[i - 1].replace('\'', '').strip()  #value
          p[colname[0]] = words[i - 2].replace('\'', '').replace('(', '').strip() #code
        plist.append(p)

        # assign codeid
        if not (p[colname[0]]) in codeid.keys():
          codeid[(p[colname[0]])] = codenum
          codenum = codenum + 1

      feature[words[0]] = plist

      if not words[0] in patientid.keys():
        patientid[words[0]] = patientnum
        patientnum += 1

      numline += 1
      #if numline > 2: break

    write_code()
    write_patient()

def read_demo():
  ''' fill demo, from demographics '''
  with open(Demographics, 'r') as f:
    numline = 0
    for line in f:
      words = line.split(',')
      words[0] = words[0].replace('(', '').replace('\'', '').strip()
      words[1] = words[1].replace('\'', '').strip()
      words[2] = words[2].replace(')', '').replace('\'', '').strip()
      demo[words[0]] = words[1:]
      numline += 1

def generate_samples():
  unique_feature = set()
  od_feature = collections.OrderedDict(sorted(feature.items()))
  with open(Labels, 'r') as f:
    numsample = 0
    maxrecord = 0
    for line in f:
      numrecord = 0
      words = line.split(',')
      pid = words[0].replace('(', '').replace('\'', '').strip()
      label = words[1].replace('\'', '').strip()
      words[2] = words[2].replace(')', '').replace('\'', '').strip()
      t = datetime.strptime(words[2], '%Y-%m-%d %H:%M:%S')
      posix_time = int(time.mktime(t.timetuple()))
      out = ''
      delta_t = 0
      for key, value in od_feature.items():
        #print "key: " + key
        #print "pid: " + pid
        if key < pid:
          continue
        elif key == pid:
          pos = 0
          for e in value:
            pos += 1
            posix_cur = int(time.mktime(e['time'].timetuple()))
            if posix_time <= posix_cur:
              break
            #if int((posix_time - posix_cur) / DAY) <= 0:
            #  break;
            
            delta_t = posix_time - posix_cur
          #print "pos: %d" %pos
          #print "delta_t: %d" %delta_t

          #flag = True # indicate a new record
          f = ''
          v = ''
          posix_pre = int(time.mktime(value[0]['time'].timetuple()))
          out += str(patientid[key]) + CtrlC + '0' + CtrlC
          for i in range(pos):
            posix_cur = int(time.mktime(value[i]['time'].timetuple()))
            #if posix_cur == posix_pre:
            if posix_cur - posix_pre == 0:
              #print value[i]['code']
              f += str(codeid[value[i]['code']]) + CtrlD
              unique_feature.add(codeid[value[i]['code']])
              v += value[i]['value'] + CtrlD
              #flag = False
            else:
              #print demo[key]
              numrecord += 1
              out += (f[:-1] + CtrlC + v[:-1] + CtrlC + demo[key][0] + CtrlC + demo[key][1]) + CtrlC + 'a' + CtrlB
              out += str(patientid[key]) + CtrlC + str(posix_cur - posix_pre) + CtrlC
              f = str(codeid[value[i]['code']]) + CtrlD
              unique_feature.add(codeid[value[i]['code']])
              v = value[i]['value'] + CtrlD
              posix_pre = posix_cur
              #flag = True

          #if flag == False:
          numrecord += 1
          out += (f[:-1] + CtrlC + v[:-1] + CtrlC + demo[key][0] + CtrlC + demo[key][1]) + CtrlC + 'a' + CtrlB
          out = out[:-1].replace('a', str(numrecord))
          #print out

          #if numrecord > maxrecord:
          #  maxrecord = numrecord

        else:
          break

      out += CtrlA + str(patientid[pid]) + CtrlB + str(delta_t) + CtrlB + label + '\n'
      #print numrecord
      if numrecord > threshold:
          continue
      numsample += 1
      if numrecord > maxrecord:
        maxrecord = numrecord
      shard.write(out)
      #if numsample > 5: break
    print "num of samples: " + str(numsample)
    print "max num of records: " + str(maxrecord)
    print "unique feature num: " + str(len(unique_feature))
    for item in unique_feature:
        print "%d" %item


cid = open('codeid', 'w')
def write_code():
  for k, v in codeid.items():
    cid.writelines(k + ', ' + str(v) + '\n')

patient_code = open('patientid', 'w')
def write_patient():
  for k, v in patientid.items():
    patient_code.writelines(k + ', ' + str(v) + '\n')



#main
if len(sys.argv) > 1:
  pnum = sys.argv[1]
else:
  pnum = 0

read_features()
read_demo()
generate_samples()
