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
print "totally %s rows in Label" % count  # the count should be 2640

conn.commit()
curs.close()
conn.close()

# I will merge part 1 and then save all of the 3 parts seperately.

# choose the smallest NIRC, Time
def find_min_idx(l):
    min = 0
    i = 1
    while i < len(l):
        if l[i][0] < l[min][0]:
            min = i
        elif l[i][0] == l[min][0] and l[i][-1] < l[min][-1]:
            min = i
        i += 1
    return min

def is_new_patient(l, p):
    for item in l:
        if item[0] <= p[0]:
            return False
    return True

def merge_by_NRIC(l1, l2, l3, l4, out):
    a, b, c, d = 0, 0, 0, 0
    observed = []
    flag = True
    observed.append(l1[a])
    observed.append(l2[b])
    observed.append(l3[c])
    observed.append(l4[d])
    while observed:
        min = find_min_idx(observed)
        if flag == True:
            out.write("\n")
            #print "min="+str(min)
            out.write(str(observed[min][0]) + "," + str(observed[min][1:]))
        else:
            out.write(',' + str(observed[min][1:]))
        patient = observed[min]
        del observed[min]
        if min == 0 and a < len(l1) - 1:
            a += 1
            observed.insert(0, l1[a])
        elif min == 1 and b < len(l2) - 1:
            b += 1
            observed.insert(1, l2[b])
        elif min == 2 and c < len(l3) - 1:
            c += 1
            observed.insert(2, l3[c])
        elif min == 3 and d < len(l4) - 1:
            d += 1
            observed.insert(3, l4[d])
        flag = is_new_patient(observed, patient)

# main
features = open("features", "w")
demo = open("demographics", "w")
labels = open("labels", "w")

# merge and write part 1
merge_by_NRIC(DIAG, LT_TOTAL, LT_ABNORMAL, MED, features)
print "Merge OK!"

# write part 2
for item in DEMO:
    demo.write(str(item).replace('L', '') + '\n')

# write part 3
# label should be in [10, 60]
for item in LABEL:
    labels.write(str(item).replace('&lt; 10', '0').replace('&gt; 60', '100') + '\n')

features.close()
demo.close()
labels.close()
