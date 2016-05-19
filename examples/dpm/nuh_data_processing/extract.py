# !/usr/bin/env python
# coding=utf-8

import MySQLdb
conn = MySQLdb.connect(host='dbpcm.d1.comp.nus.edu.sg',
                       user='sqlsugg',
                       passwd='sqlsugg')
curs = conn.cursor()

curs.execute("use DPM_AND_CKD")

# 1. time-denpendent feature part
count = curs.execute(
    "select NRIC, ICD10_UNIFY, '1' as value, DiagTime from DM_CKD1_2640_DIAG_INTERMEDIATE_2 group by NRIC, ICD10_UNIFY, DiagTime order by NRIC, DiagTime")
DIAG = curs.fetchall()
print "totally %s rows in DIAGNOSIS" % count

count = curs.execute(
    "select NRIC, LabTestCode, NB_FindingCode, FindingDateTime from DM_CKD1_2640_LAB_INTERMEDIATE_2_TOTAL order by NRIC, FindingDateTime")
LT_TOTAL = curs.fetchall()
print "totally %s rows in labtest_total_count" % count

count = curs.execute(
    "select NRIC, LabTestCode, NB_ABFindingCode, FindingDateTime from DM_CKD1_2640_LAB_INTERMEDIATE_2_ABNORMAL order by NRIC, FindingDateTime")
LT_ABNORMAL = curs.fetchall()
print "totally %s rows in labtest_abnormal_count" % count

count = curs.execute(
    "select NRIC, MedDetailsMedName, '1' as value, CompletedDate from DM_CKD1_2640_MED_INTERMEDIATE_2 group by NRIC, MedDetailsMedName, CompletedDate order by NRIC, CompletedDate")
MED = curs.fetchall()
print "totally %s rows in MEDICATION" % count

# 2. time-independent feature part
count = curs.execute(
    "select NRIC, Age, Gender from DM_CKD1_2640_DEMO order by NRIC")
DEMO = curs.fetchall()
print "totally %s rows in DEMOGRAPHICS" % count  # the count should be 2640

# 3. label part
count = curs.execute(
    "select NRIC, FindingAmount, FindingDateTime from DM_CKD1_2640_LAB_ALL_FIELDS"
    + " where (LabTestCode = 'EGFRF' or LabTestCode = 'EGFRC')"
    + " and (FindingAmount >= 10 and FindingAmount <= 60)"
    + " group by NRIC, FindingDateTime order by NRIC, FindingDateTime")
LABEL = curs.fetchall()
print "totally %s rows in Label" % count

conn.commit()
curs.close()
conn.close()

# I will merge part 1 and then save all of the 3 parts seperately.

# choose the smallest (NIRC, Time) pair
def find_min_idx(l):
    min = 0
    i = 1
    idx = 0
    while idx < len(l): # find the first non-minus-one index
        if l[min] == -1:
            min += 1;
        else: break;

    while i < len(l): # find the smallest among "observed", typically should have 4 items
        #print l[i]
        #print (l[i] == -1)
        if l[i] == -1:
            i += 1;
            continue;
        if l[i][0] < l[min][0]: # order according to NRIC information
            min = i
        elif l[i][0] == l[min][0] and l[i][-1] < l[min][-1]: # NRIC is the same, order according to time information
            #print "---"
            #print l[i][0], l[i][-1]
            #print l[min][0], l[min][-1]
            min = i
        i += 1
    return min

def is_new_patient(l, p):
    for item in l:
        if item == -1:
            continue;
        if item[0] <= p[0]: # if any information in "observed" is still the NRIC, then still the old patient
            return False
    #print "---New Patient---"
    #print l
    #print p
    return True

def merge_by_NRIC(l1, l2, l3, l4, out):
    a, b, c, d = 0, 0, 0, 0 # corresponding to the index for DIAG, LT_TOTAL, LT_ABNORMAL, MED
    observed = [] # during the processing, the size of "observed" should always be 4
    flag = True
    observed.append(l1[a])
    observed.append(l2[b])
    observed.append(l3[c])
    observed.append(l4[d])
    while not (a == len(l1) - 1 and b == len(l2) - 1 and c == len(l3) - 1 and d == len(l4) - 1): # will be empty if all files have been processed
        min = find_min_idx(observed)
        if flag == True: # for a new patient, also need to print NRIC info
            out.write("\n")
            #print "min="+str(min)
            out.write(str(observed[min][0]) + "," + str(observed[min][1:]))
        else: # for the current patient, print medical features
            out.write(',' + str(observed[min][1:]))
        patient = observed[min] # a record with current NRIC info from DIAG, LAB_TOTAL, LAB_ABNORMAL, MED
        del observed[min]
        if min == 0:
            if a < len(l1) - 1: # if already record the information, then need to make up using new one
                a += 1
                observed.insert(0, l1[a])
            else: observed.insert(0, -1) # if this category of features already ends, then use "-1"
        elif min == 1:
            if b < len(l2) - 1:
                b += 1
                observed.insert(1, l2[b])
            else:
                observed.insert(1, -1)
        elif min == 2:
            if c < len(l3) - 1:
                c += 1
                observed.insert(2, l3[c])
            else: observed.insert(2, -1)
        elif min == 3 :
            if d < len(l4) - 1:
                d += 1
                observed.insert(3, l4[d])
            else: observed.insert(3, -1)
        flag = is_new_patient(observed, patient)
        #print "Observed"
        #print observed
    out.write(',' + str(observed[min][1:]))

# main
features = open("features", "w")
demo = open("demographics", "w")
labels = open("labels", "w")

# merge and write part 1
merge_by_NRIC(DIAG, LT_TOTAL, LT_ABNORMAL, MED, features) # write time-dependent features into "features"
print "Merge OK!"

# write part 2
for item in DEMO:
    demo.write(str(item).replace('L', '') + '\n') # This is because in the demo features, there may be "BLOB" format information which contains "L"

# write part 3
# label should be in [10, 60]
for item in LABEL:
    labels.write(str(item).replace('&lt; 10', '0').replace('&gt; 60', '100') + '\n') # actually no need as when we extract from LAB TEST table, already constrain the GFR in [10,60]

features.close()
demo.close()
labels.close()
