import sys
import numpy as np

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
    test_ratio = 0.1
    test_index = 1 # The index starts from 1
    label_feature = 'MMSCORE'
    test_file_path = 'test.shard'
    train_file_path = 'train.shard'

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
    for viscode in viscode_set:
        try:
            assert viscode
            assert 'm' == viscode[0]
            num = int(viscode[1:])
            viscode_dict_num[viscode] = num
        except (AssertionError, ValueError):
            pass

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
            if viscode in viscode_dict_num and viscode_dict_num[viscode] < cut_point:
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
    for rid in sorted(constant_features.keys(), key=lambda x: int(x)):
        pre_cut = []
        post_cut = []
        for viscode in data[rid]:
            if viscode in viscode_dict_num:
                viscode_num = viscode_dict_num[viscode]
                if viscode_num < cut_point:
                    pre_cut.append((viscode, viscode_num))
                else:
                    post_cut.append((viscode, viscode_num))

        # sort viscode by viscode num
        pre_cut.sort(key=lambda x: x[1])
        post_cut.sort(key=lambda x: x[1])

        # prepare input part
        pre_viscode_num = 0
        observed_idx = []
        feature_value = []
        input_records = []
        for viscode, viscode_num in pre_cut:
            lap_time = str(viscode_num - pre_viscode_num)
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
            norm_const_fea = constant_features[rid]
            for idx in range(len(constant_fea_name)):
                f_mean, f_std = fea_mean_std[constant_fea_name[idx]]
                f_val = (norm_const_fea[idx] - f_mean) / f_std if abs(f_std) > np.finfo(np.float32).eps else 0.0
                norm_const_fea[idx] = f_val
            norm_const_fea = [str(item) for item in norm_const_fea]

            record_items = [rid, lap_time, ctrlD.join(observed_idx), ctrlD.join(feature_value)] + norm_const_fea
            input_records.append(ctrlC.join(record_items))
        input_part = ctrlB.join(input_records)

        # prepare label part
        labels = []
        for viscode, viscode_num in post_cut:
            try:
                label = str(float(data[rid][viscode][label_feature]))
                delta_time = str(viscode_num - pre_viscode_num)
                label_items = [rid, delta_time, label]
                labels.append(ctrlB.join(label_items))
            except (KeyError, ValueError):
                pass

        for label in labels:
            sample_lines.append(ctrlA.join([input_part, label]))

    # write output file
    total_sample = len(sample_lines)
    test_start = (test_index - 1) * int(total_sample * test_ratio)
    test_end = test_start + int(total_sample * test_ratio)
    test_end = min(total_sample, test_end)

    # write test shard file
    with open(test_file_path, 'w') as f:
        f.writelines(sample_lines[idx] + '\n' for idx in range(test_start, test_end))

    # write train shard file
    with open(train_file_path, 'w') as f:
        f.writelines(sample_lines[idx] + '\n' for idx in range(total_sample) if not test_start <= idx < test_end)
