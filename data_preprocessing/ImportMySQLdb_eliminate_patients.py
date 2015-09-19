import MySQLdb


def build_item_name_index():
    ori_sql = 'select distinct {0} from {1}'

    for item_name, code_dict, col_name, table_name, info_dict in sql_items:
        cursor.execute(ori_sql.format(col_name, table_name))
        for col, in cursor:
            idx = len(code_dict)
            code_dict.setdefault(col.strip(), idx)


def dump_data():
    group_count_sql_temp = \
        'select NRIC, {0}, count(*) as Freq from {1} where TimeSpan between {2} and {3} group by NRIC, {4};'

    test_fail_sql = "select NRIC, LabTestCode, count(*) as Freq from DM_CKD1_2640_LAB_DERIVE where TimeSpan between " \
                    "{0} and {1} and FindingCode in ('H', 'L') group by NRIC, LabTestCode;"

    for i in range(shard_num):
        start_time, end_time = shard_interval[i]
        offset = 0
        # diag, test, med, proc processing are same
        for item_name, code_dict, col_name, table_name, info_dict in sql_items:
            cursor.execute(group_count_sql_temp.format(col_name, table_name, start_time, end_time, col_name))
            for nric, code, freq in cursor:
                info_dict.setdefault(nric,
                                     [[] for _ in range(shard_num)])[i].append((code_dict[code.strip()] + offset, freq))
            offset += len(code_dict)

        # test fail needs special sql logic
        cursor.execute(test_fail_sql.format(start_time, end_time))
        for nric, code, freq in cursor:
            test_fail_info_dict.setdefault(nric, [[] for _ in range(shard_num)])[i].append(
                (test_code_dict[code.strip()] + offset, freq))

    # dump ground truth data
    truth_sql = 'select NRIC, Label from DM_CKD1_2640_GROUND_TRUTH_DERIVE'
    cursor.execute(truth_sql)
    for nric, label in cursor:
        truth_info_dict[nric] = label


def format_vector(tuple_list):
    return '\003'.join(['{0}\004{1}'.format(a, b) for a, b in tuple_list])


def calc_feature_size(nric):
    cnt = 0
    for shard_idx in range(shard_num):
        cnt += len(diag_info_dict.setdefault(nric, [[] for _ in range(shard_num)])[shard_idx])
        cnt += len(test_info_dict.setdefault(nric, [[] for _ in range(shard_num)])[shard_idx])
        cnt += len(test_fail_info_dict.setdefault(nric, [[] for _ in range(shard_num)])[shard_idx])
        cnt += len(med_info_dict.setdefault(nric, [[] for _ in range(shard_num)])[shard_idx])
    return cnt


def remove_empty_patients(input_dict):
    output_dict = dict()
    for nric, label in input_dict.iteritems():
        if calc_feature_size(nric):
            output_dict.setdefault(nric, label)
    return output_dict


def format_output(nric, label):
    feature_shard = []
    for shard_idx in range(shard_num):
        feature_shard.append('\003'.join([
            format_vector(diag_info_dict.setdefault(nric, [[] for _ in range(shard_num)])[shard_idx]),
            format_vector(test_info_dict.setdefault(nric, [[] for _ in range(shard_num)])[shard_idx]),
            format_vector(test_fail_info_dict.setdefault(nric, [[] for _ in range(shard_num)])[shard_idx]),
            format_vector(med_info_dict.setdefault(nric, [[] for _ in range(shard_num)])[shard_idx])
        ]))
    return '\001'.join([str(nric)] + feature_shard + [str(int(label)-2)]) + '\n'


def print_output():
    with open(output_file_path, 'w') as f:
        # line 1: number of unique NRIC
        f.write('{0}\n'.format(len(truth_info_dict)))
        # line 2: number of shard
        f.write('{0}\n'.format(shard_num))
        # line 3: dimension of features
        f.write('{0}\n'.format(
            len(diag_code_dict) + len(test_code_dict) * 2 + len(med_desc_dict)))
        # data seg
        for nric, label in truth_info_dict.iteritems():
            f.write(format_output(nric, label))


def print_code_dict():
    with open('code_dict.txt', 'w') as f:
        offset = 0
        for item_name, code_dict, col_name, table_name, info_dict in sql_items:
            for code, idx in code_dict.iteritems():
                f.write('{0}: {1} [{2}]\n'.format(item_name, code, idx + offset))
            offset += len(code_dict)

        # special logic for lab test fail
        for code, idx in test_code_dict.iteritems():
            f.write('{0} [{1}]: {2}\n'.format('labtestfail', code, idx + offset))


if __name__ == '__main__':
    # configuration of db connection
    conn = MySQLdb.connect(
        host='hostname',
        user='user',
        passwd='password',
        db='DPM_AND_CKD',
        charset='utf8')
    cursor = conn.cursor()

    # attr of shard generation
    sec_per_day = 24 * 3600
    attr_a = 360 * sec_per_day
    attr_b = 180 * sec_per_day
    attr_w = 120 * sec_per_day
    assert not attr_a % attr_w  # attr_a can be divide by attr_w exactly

    # output file conf
    output_file_path = 'output.txt'

    # calculate time span intervals for each shard
    shard_num = attr_a / attr_w
    shard_interval = [(attr_b + attr_w * i, attr_b + attr_w * i + attr_w - 1) for i in range(shard_num)[::-1]]

    # code/desc index dicts
    diag_code_dict = dict()
    test_code_dict = dict()
    med_desc_dict = dict()

    # formatted information
    diag_info_dict = dict()
    test_info_dict = dict()
    test_fail_info_dict = dict()
    med_info_dict = dict()
    truth_info_dict = dict()

    sql_items = [
        ('diag', diag_code_dict, 'ICD10_UNIFY', 'DM_CKD1_2640_DIAG_DERIVE', diag_info_dict),
        ('labtest', test_code_dict, 'LabTestCode', 'DM_CKD1_2640_LAB_DERIVE', test_info_dict),
        ('med', med_desc_dict, 'MedDetailsMedName', 'DM_CKD1_2640_MED_DERIVE', med_info_dict)
    ]

    # build index for multivariate items
    build_item_name_index()

    # dump and format data
    dump_data()

    # process data: step1 remove items without features
    truth_info_dict = remove_empty_patients(truth_info_dict)

    # generate output file
    print_output()

    conn.close()

    print_code_dict()
