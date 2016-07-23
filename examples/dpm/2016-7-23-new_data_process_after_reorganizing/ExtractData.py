import MySQLdb
import sys
import random

RELATIVE_CUT_POINT = 3600 * 24 * 30 * 12 # Relative cutpoint(if need finer info: use 3600*24*365), Absolute should be StartTimePoint + RelativeTime for each patient
GRANUALARITY = 3600 * 24 # DAY as the window size
#GRANUALARITY = 3600 * 24 * 7 # Week as the window size
#GRANUALARITY = 3600 * 24 * 30  Month as the window size

ctrlA = '\001'
ctrlB = '\002'
ctrlC = '\003'
ctrlD = '\004'

# ratio separator between train & valid, valid & test respectively (currently not use validation data)
ratio_train_valid = 0.9
ratio_valid_test = 0.9

# directory of shard-format files
test_file_path = 'test.shard'
valid_file_path = 'valid.shard'
train_file_path = 'train.shard'

# parse time in string-format into [seconds] after epoch (year 1970)
def parse_time(time_str):
    import time
    return int(time.mktime(time.strptime(time_str,'%Y-%m-%d %H:%M:%S')))

if __name__ == '__main__':
    print "relative cutpoint: ", RELATIVE_CUT_POINT
    print "granularity: ", GRANUALARITY

    ### Step1. Extract all data from DB and organize data into dictionary format
    # connect to db
    conn = MySQLdb.connect(host='dbpcm.d1.comp.nus.edu.sg',
                           user='XXX',
                           passwd='XXX')
    cur = conn.cursor()
    cur.execute("use DPM_AND_CKD")

    nric_starttime = dict() # NRIC -> start time of all his medical features (Diag, Lab_Total, Lab_Abnormal, Med)
    nric_recordnum = dict() # NRIC -> record number before cutpoint

    # grab demographic info: 2 dict respectively: NRIC -> age; NRIC -> gender
    nric_age = dict() # NRIC -> age
    nric_gender = dict() # NRIC -> gender
    cur.execute('SELECT NRIC, Age, Gender from DM_CKD1_2640_DEMO;')
    result = cur.fetchall()
    for line in result:
        nric = line[0]
        nric_age[nric] = line[1]
        nric_gender[nric] = line[2]
    print "Demo features finished"

    #[TODO] kaiping: later can consider UTC+8 (our data) VS UTC+0 (1970 starting epoch) if needed
    # the reason for processing 4 medical feature tables respectively instead of jointly: the processing logic of cnt values is different
    # grab diagnosis info: 1 dict: NRIC -> time -> diagcode -> cnt; information with the most detailed granularity --> transform into unit of GRANULARITY
    nric_time_diagcode_count = dict()
    cur.execute('SELECT NRIC, ICD10_UNIFY, DiagTime from DM_CKD1_2640_DIAG_INTERMEDIATE_2;')
    result = cur.fetchall()
    for line in result:
        nric = line[0]
        code = line[1]
        time = parse_time(line[2]) / GRANUALARITY # transform the absolute time from [seconds] into [days]
        tmp = nric_starttime.setdefault(nric, sys.maxint)
        nric_starttime[nric] = min(tmp, time) # obtain the minimum start timepoint [days] among all medical features for each patient
        nric_time_diagcode_count.setdefault(nric, dict()).setdefault(time, dict()).setdefault(code, 0.0)
        nric_time_diagcode_count[nric][time][code] += 1
    print "Diagnosis features finished"

    # grab medication info: 1 dict: NRIC -> time -> medcode -> cnt; information with the most detailed granularity
    nric_time_medcode_count = dict()
    cur.execute('SELECT NRIC, MedDetailsMedName, CompletedDate from DM_CKD1_2640_MED_INTERMEDIATE_2;')
    result = cur.fetchall()
    for line in result:
        nric = line[0]
        code = line[1]
        time = parse_time(line[2]) / GRANUALARITY # transform the absolute time from [seconds] into [days]
        tmp = nric_starttime.setdefault(nric, sys.maxint)
        nric_starttime[nric] = min(tmp, time) # obtain the minimum start timepoint [days] among all medical features for each patient
        nric_time_medcode_count.setdefault(nric, dict()).setdefault(time, dict()).setdefault(code, 0.0)
        nric_time_medcode_count[nric][time][code] += 1
    print "Medication features finished"

    # grab lab total info: 1 dict: NRIC -> time -> lab -> totalcnt; information with the most detailed granularity
    total_nric_time_lab_count = dict()
    cur.execute('select NRIC, LabTestCode, FindingDateTime, NB_FindingCode from DM_CKD1_2640_LAB_INTERMEDIATE_2_TOTAL;')
    result = cur.fetchall()
    for line in result:
        nric = line[0]
        code = line[1]
        time = parse_time(line[2]) / GRANUALARITY # transform the absolute time from [seconds] into [days]
        tmp = nric_starttime.setdefault(nric, sys.maxint)
        nric_starttime[nric] = min(tmp, time) # obtain the minimum start timepoint [days] among all medical features for each patient
        count = int(line[3])
        total_nric_time_lab_count.setdefault(nric, dict()).setdefault(time, dict()).setdefault(code, 0.0)
        total_nric_time_lab_count[nric][time][code] += count
    print "Lab test total features finished"

    # grab lab abnormal info: 1 dict: NRIC -> time -> @lab -> abnormalcnt; information with the most detailed granularity
    abnormal_nric_time_lab_count = dict()
    cur.execute(
        'select NRIC, LabTestCode, FindingDateTime, NB_ABFindingCode from DM_CKD1_2640_LAB_INTERMEDIATE_2_ABNORMAL;')
    result = cur.fetchall()
    for line in result:
        nric = line[0]
        code = line[1]
        time = parse_time(line[2]) / GRANUALARITY # transform the absolute time from [seconds] into [days]
        tmp = nric_starttime.setdefault(nric, sys.maxint)
        nric_starttime[nric] = min(tmp, time) # obtain the minimum start timepoint [days] among all medical features for each patient
        count = int(line[3])
        abnormal_nric_time_lab_count.setdefault(nric, dict()).setdefault(time, dict()).setdefault(code, 0.0)
        abnormal_nric_time_lab_count[nric][time][code] += count
    print "Lab test abnormal features finished"

    # grab label information: 1 dict: NRIC -> time -> label_value (average value if more than 1 same lab test in the same time)
    label_nric_time_value = dict()
    label_nric_time_count = dict() # for average
    cur.execute(
        'select NRIC, FindingAmount, FindingDateTime from DM_CKD1_2640_LAB_ALL_FIELDS where LabTestCode = \'EGFRF\' or LabTestCode = \'EGFRC\';')
    result = cur.fetchall()
    for line in result:
        nric = line[0]
        time = parse_time(line[2]) / GRANUALARITY # should also be in [day] unit for prediction
        if time < (nric_starttime[nric] + (RELATIVE_CUT_POINT / GRANUALARITY)):
            continue
        try: # otherwise (only keep the label information after cutpoint)
            value = float(line[1])
            assert value >= 10.0
            assert value <= 60.0
        except (AssertionError, ValueError):
            continue
        pre_count = label_nric_time_count.setdefault(nric, dict()).setdefault(time, 0.0)
        pre_value = label_nric_time_value.setdefault(nric, dict()).setdefault(time, 0.0)
        label_nric_time_count[nric][time] += 1
        label_nric_time_value[nric][time] = (value + pre_count * pre_value) / (pre_count + 1)
    print "Label features finished"
    print "Patient number for generating samples: ", len(label_nric_time_value) # Note this is the number of patients who can generate samples, but possibly no features

    ### Step2. Transform 4 categories of medical features into 1 dict() and Normalization

    # organize into NRIC -> Time -> feature_code -> feature_value (count)
    feature_dicts = [
        nric_time_diagcode_count,
        nric_time_medcode_count,
        total_nric_time_lab_count,
        abnormal_nric_time_lab_count
    ]

    nric_time_featurecode_count = dict()
    feature_exist = set() # for organizing features appearing before cutpoint and indexing them
    for nric in nric_age:
        nric_timeset = set()
        for feature_dict in feature_dicts:
            for time in feature_dict[nric].keys(): # for each category of features, for each day, need to aggregate (the same feature may appear multiple times in the same GRANULARITY)
                if time < (nric_starttime[nric] + (RELATIVE_CUT_POINT / GRANUALARITY)): # only add the features before cutpoint
                    nric_timeset.add(time)
                    for code in feature_dict[nric][time].keys():
                        nric_time_featurecode_count.setdefault(nric, dict()).setdefault(time, dict()).setdefault(code, 0.0)
                        nric_time_featurecode_count[nric][time][code] += feature_dict[nric][time][code]
                        feature_exist.add(code)
        nric_recordnum[nric] = len(nric_timeset)
        print "Number of records before CP for patient ", nric, " ", nric_recordnum[nric]
    print "feature num: ", len(feature_exist) # 5082, need to check again

    # count the maximum count value for each feature
    # feature_code -> maximum count value
    feature_max = dict()
    for nric in nric_age:
        for time in nric_time_featurecode_count[nric].keys():
            for code in nric_time_featurecode_count[nric][time].keys():
                cur_value = nric_time_featurecode_count[nric][time][code]
                pre_value = feature_max.setdefault(code, 0.0)
                feature_max[code] = max(pre_value, cur_value)

    # encode all medical features with index
    feature_idx = dict()
    for code, idx in zip(feature_exist, range(len(feature_exist))):
        feature_idx[code] = idx


    ### Step3. Write to shard-format file
    sample_lines = []
    for nric in nric_time_featurecode_count:
        if nric not in label_nric_time_value:
            print "patient: ", nric, " only have feature input, but not have label information"
            continue
        print "Used patient: ", nric, "have both feature input and label information"
        # for patients have both feature input and label output
        # 1) prepare input part
        # prepare time-dependent features (3 time-indepedent fetaures (age, gender, recordnum already in three dict()))
        input_records = [] # one record corresponds to one [day] combining all features in that day
        observed_idx = []
        feature_value = []
        for cur_time in sorted(nric_time_featurecode_count[nric].keys(), key=lambda x:int(x)): # iterate time (with the features for one patient) in time order; not order the dict
            pre_last_time = nric_starttime[nric]
            lap_time = cur_time - pre_last_time
            assert lap_time >= 0
            pre_last_time = cur_time # absolute time point, at last will be the last timepoint before cutpoint

            for code in nric_time_featurecode_count[nric][cur_time].keys():
                observed_idx.append(str(feature_idx[code]))
                feature_value.append(str(float(nric_time_featurecode_count[nric][cur_time][code] / (1.0 * feature_max[code]))))

            # the last three are constant features for each patient
            record_items = [nric, str(lap_time), ctrlD.join(observed_idx), ctrlD.join(feature_value), str(nric_age[nric]), str(nric_gender[nric]), str(nric_recordnum[nric])]
            input_records.append(ctrlC.join(record_items)) # add one record (corresponding to one time)
        input_part = ctrlB.join(input_records)

        # 2) prepare label part (all samples for one patient should have the same input part)
        labels = []
        for time in label_nric_time_value[nric].keys(): # no need for ordering time, all samples are independent of each other
            delta_time = str(time - pre_last_time)
            label = str(label_nric_time_value[nric][time])
            label_items = [nric, delta_time, label]
            labels.append(ctrlB.join(label_items)) # add one label corresponding to one sample

        for label in labels:
            sample_lines.append(ctrlA.join([input_part, label])) # one sample

    ###Step4. write to output file and shuffle the data
    total_sample = len(sample_lines)

    # select random subsection as train data, valida data and test data respectively
    random.shuffle(sample_lines)
    split_train_valid = int(total_sample * ratio_train_valid)
    split_valid_test = int(total_sample * ratio_valid_test)
    train_data = sample_lines[:split_train_valid]
    valid_data = sample_lines[split_train_valid:split_valid_test]
    test_data = sample_lines[split_valid_test:]

    print "number of training samples: ", len(train_data)
    print "number of validation samples: ", len(valid_data)
    print "number of testing samples: ", len(test_data)

    # write to shard-format files
    with open(train_file_path, 'w') as f:
        f.writelines(train_data[idx] + '\n' for idx in range(len(train_data)))

    with open(valid_file_path, 'w') as f:
        f.writelines(valid_data[idx] + '\n' for idx in range(len(valid_data)))

    with open(test_file_path, 'w') as f:
        f.writelines(test_data[idx] + '\n' for idx in range(len(test_data)))