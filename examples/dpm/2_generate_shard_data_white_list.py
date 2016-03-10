import sys
import numpy as np
import random

priority_m00 = ['bl', 'sc']
ignore_features = ['PTDOBMM', 'PTDOBYY', 'USERDATE', 'PTEDUCAT', 'PTGENDER']
constant_fea_name = ['AGE', 'EDUCATION', 'GENDER']

[RID, TIME, FEATURE_NAME, FEATURE_VALUE] = range(4)
[BIRTH_M, BIRTH_Y, USERDATE, EDU, GENDER] = range(5)

ctrlA = '\001'
ctrlB = '\002'
ctrlC = '\003'
ctrlD = '\004'


def seek_supreme_feature(feature, viscode_dict_num, datum):
    """
    seek a feature's value among multi viscodes, return the supreme value ordered by viscode num.

    Args:
        feature: feature name
        viscode_dict_num: dict(viscode -> priority num), lower is prior
        datum: dict: time -> fea_name -> fea_value

    Returns:
        value if valid, None otherwise
    """
    ret = None
    cur_viscode = sys.maxint
    for viscode in datum:
        if viscode in viscode_dict_num and feature in datum[viscode] and viscode_dict_num[viscode] < cur_viscode:
            cur_viscode = viscode_dict_num[viscode]
            ret = datum[viscode][feature]
    return ret


if __name__ == '__main__':
    # input parameters
    csv_file_path = 'ckd_1year/TEST_REMOVE_EMPTY.csv'
    cut_point = 13
    #cut_point = 25 # For a second experimental setting

    test_ratio = 0.1
    #valid_ratio = 0.09
    valid_ratio = 0 # Only use training data and testing data in this experimental setting
    test_index = 1  # Pick the 1st part of data as testing data -> For more fair comparison

    label_feature = 'MMSCORE'
    test_file_path = 'test.shard'
    valid_file_path = 'valid.shard'
    train_file_path = 'train.shard'

    # only consider those viscode in white list, None means all valid viscodes beside cutpoint.
    #input_white_list = None
    #output_white_list = None

    #### Experimental Setting 1 - 7 time points

    ## Sub-Setting 1 - cutpoint = M13
    # 1)
    input_white_list = ['m12']
    # 2)
    #input_white_list = ['m06', 'm12']
    # 3)
    #input_white_list = ['m00', 'm06', 'm12']

    ## Sub-Setting 2 - cutpoint = M25
    # 1)
    #input_white_list = ['m24']
    # 2)
    #input_white_list = ['m18', 'm24']
    # 3)
    #input_white_list = ['m12','m18', 'm24']
    # 4)
    #input_white_list = ['m06', 'm12','m18', 'm24']
    # 5)
    #input_white_list = ['m00', 'm06', 'm12','m18', 'm24']

    output_white_list = None

    #### Experimental Setting 2 - 6 time points

    ## Sub-Setting 1 - cutpoint = M13
    # 1)
    #input_white_list = ['m12']
    # 2)
    #input_white_list = ['m06', 'm12']
    # 3)
    #input_white_list = ['m00', 'm06', 'm12']

    #output_white_list= ['m24', 'm36', 'm48']

    ## Sub-Setting 2 - cutpoint = M25
    # 1)
    #input_white_list = ['m24']
    # 2)
    #input_white_list = ['m12', 'm24']
    # 3)
    #input_white_list = ['m06', 'm12', 'm24']
    # 4)
    #input_white_list = ['m00', 'm06', 'm12', 'm24']

    #output_white_list= ['m36', 'm48']


    # option for lap time normalization
    #lap_time_norm = True
    lap_time_norm = False # Not normalize time-related features in this experimental setting
    #include_delta_time = True
    include_delta_time = False

    # load csv data
    with open(csv_file_path, 'r') as f:
        raw_data = np.array([[item.strip('\"').strip() for item in line.strip().split(',')]
                             for line in f.readlines()[1:]])

    # dict: rid -> time -> feature -> value
    data = {}
    feature_set = set()
    viscode_set = set()
    for datum in raw_data:
        data.setdefault(datum[RID], dict()).setdefault(datum[TIME], dict())[datum[FEATURE_NAME]] = datum[FEATURE_VALUE]
        viscode_set.add(datum[TIME])
        feature_set.add(datum[FEATURE_NAME])
    feature_set -= set(ignore_features)

    # construct m00 viscode
    for rid in data:
        # loop by revsersed order
        for viscode in reversed(priority_m00):
            for k, v in data[rid].setdefault(viscode, dict()).iteritems():
                data[rid].setdefault('m00', dict())[k] = v
    viscode_set.add('m00')

    # parse viscode into integer
    viscode_dict_num = dict()
    pre_cut_viscode = []
    post_cut_viscode = []
    for viscode in viscode_set:
        try:
            assert viscode
            assert 'm' == viscode[0]
            num = int(viscode[1:])
            viscode_dict_num[viscode] = num
            if num < cut_point:
                pre_cut_viscode.append(viscode)
            else:
                post_cut_viscode.append(viscode)
        except (AssertionError, ValueError):
            pass

    # generate default input/output viscode white list
    if input_white_list is None:
        input_white_list = pre_cut_viscode
    input_white_list = set(input_white_list)
    if output_white_list is None:
        output_white_list = post_cut_viscode
    output_white_list = set(output_white_list)

    # prepare constant features
    features = dict()
    constant_features = dict()
    for rid in data:
        try:
            special_field = [seek_supreme_feature(feature, viscode_dict_num, data[rid]) for feature in ignore_features]
            # calc age
            cur_date = special_field[USERDATE].split('-')
            cur_yea = float(cur_date[0])
            cur_mon = float(cur_date[1])
            bir_mon = float(special_field[BIRTH_M])
            bir_yea = float(special_field[BIRTH_Y])

            age = cur_yea - bir_yea + (cur_mon - bir_mon) / 12
            edu = float(special_field[EDU])
            sex = float(special_field[GENDER])
            constant_features[rid] = [age, edu, sex]
            features.setdefault('AGE', []).append(age)
            features.setdefault('EDUCATION', []).append(edu)
            features.setdefault('GENDER', []).append(sex)
        except (AttributeError, TypeError):
            pass

    # collect all chosen feature for normalization
    for rid in constant_features:
        f_cnt = 0  # this variable name inherit from the orginal
        for viscode in data[rid]:
            if viscode in input_white_list:
                f_cnt += 1
                for k, v in data[rid][viscode].iteritems():
                    if k not in ignore_features:
                        try:
                            features.setdefault(k, []).append(float(v))
                        except ValueError:
                            print 'WARN: value parse failed. {}'.format(str((rid, viscode, k, v)))
        constant_features[rid].append(f_cnt)
        features.setdefault('F_CNT', []).append(f_cnt)

    # calc mean and std for each chosen feature
    fea_mean_std = dict()
    for k in features:
        assert len(features[k]) > 0
        if np.abs(np.std(features[k])) < np.finfo(np.float32).eps:
            print 'WARN: feature[{}] is constant!'.format(k)

        fea_mean_std[k] = np.mean(features[k]), np.std(features[k])

    # build feature name index system
    feature_idx = dict()
    for feature in feature_set:
        feature_idx[feature] = len(feature_idx)

    # generate records by patients with intact constant features
    sample_lines = []
    lap_delta_times = []
    lap_time_each_sample = []
    delta_time_each_sample = []
    for rid in sorted(constant_features.keys(), key=lambda x: int(x)):
        pre_cut = []
        post_cut = []
        for viscode in data[rid]:
            if viscode in viscode_dict_num:
                viscode_num = viscode_dict_num[viscode]
                if viscode in input_white_list:
                    pre_cut.append((viscode, viscode_num))
                elif viscode in output_white_list:
                    post_cut.append((viscode, viscode_num))

        # sort viscode by viscode num
        pre_cut.sort(key=lambda x: x[1])
        post_cut.sort(key=lambda x: x[1])

        # prepare input part
        pre_viscode_num = 0
        input_records = []

        # prepare constant features
        norm_const_fea = constant_features[rid]
        for idx in range(len(constant_fea_name)):
            f_mean, f_std = fea_mean_std[constant_fea_name[idx]]
            f_val = (norm_const_fea[idx] - f_mean) / f_std if abs(f_std) > np.finfo(np.float32).eps else 0.0
            norm_const_fea[idx] = f_val
        norm_const_fea = [str(item) for item in norm_const_fea]

        lap_times = []
        for viscode, viscode_num in pre_cut:
            observed_idx = []
            feature_value = []
            lap_time = float(viscode_num - pre_viscode_num)
            lap_delta_times.append(lap_time)
            lap_times.append(lap_time)
            lap_time = '{}'
            pre_viscode_num = viscode_num
            for k, v in data[rid][viscode].iteritems():
                if k in feature_idx:
                    try:
                        f_mean, f_std = fea_mean_std[k]
                        f_val = (float(v) - f_mean) / f_std if abs(f_std) > np.finfo(np.float32).eps else 0.0
                        feature_value.append(str(float(f_val)))
                        observed_idx.append(str(feature_idx[k]))
                    except ValueError:
                        pass

            record_items = [rid, lap_time, ctrlD.join(observed_idx), ctrlD.join(feature_value)] + norm_const_fea
            input_records.append(ctrlC.join(record_items))
        input_part = ctrlB.join(input_records)

        # prepare label part
        labels = []
        delta_times = []
        for viscode, viscode_num in post_cut:
            try:
                label = str(float(data[rid][viscode][label_feature]))
                delta_time = float(viscode_num - pre_viscode_num)
                delta_times.append(delta_time)
                if include_delta_time:
                    lap_delta_times.append(delta_time)
                delta_time = '{}'
                label_items = [rid, delta_time, label]
                labels.append(ctrlB.join(label_items))
            except (KeyError, ValueError):
                pass

        for label, delta_time in zip(labels, delta_times):
            sample_lines.append(ctrlA.join([input_part, label]))
            lap_time_each_sample.append(lap_times)
            delta_time_each_sample.append(delta_time)

    # lap delta time normalization
    lap_time_each_sample = [np.array(i) for i in lap_time_each_sample]
    if lap_time_norm:
        mean_time = np.mean(lap_delta_times)
        std_time = np.std(lap_delta_times)
        for idx in range(len(lap_time_each_sample)):
            lap_time_each_sample[idx] = (lap_time_each_sample[idx] - mean_time) / std_time
        if include_delta_time:
            delta_time_each_sample = (np.array(delta_time_each_sample) - mean_time) / std_time

    total_sample = len(sample_lines)
    for idx in range(total_sample):
        args = lap_time_each_sample[idx].tolist() + [delta_time_each_sample[idx]]
        sample_lines[idx] = sample_lines[idx].format(*args)

    # distribute samples for train, valid and test
    if test_index == -1:
        test_sample_set = set(random.sample(range(total_sample), int(total_sample * test_ratio)))
    else:
        test_start = (test_index - 1) * int(total_sample * test_ratio)
        test_end = test_start + int(total_sample * test_ratio)
        test_end = min(total_sample, test_end)
        test_sample_set = set(range(test_start, test_end))

    # if test index does not equals to -1, valid mode disable
    valid_mode = True if test_index == -1 else False
    if valid_mode:
        valid_num = int(total_sample * valid_ratio)
        valid_sample_set = set(random.sample(set(range(total_sample)) - test_sample_set, valid_num))
    else:
        valid_sample_set = set()

    train_sample_set = set(range(total_sample)) - test_sample_set - valid_sample_set

    # write test shard file
    with open(test_file_path, 'w') as f:
         f.writelines(sample_lines[idx] + '\n' for idx in test_sample_set)

    # write valid shard file
    if valid_mode:
        with open(valid_file_path, 'w') as f:
            f.writelines(sample_lines[idx] + '\n' for idx in valid_sample_set)

    # write train shard file
    with open(train_file_path, 'w') as f:
        f.writelines(sample_lines[idx] + '\n' for idx in train_sample_set)
