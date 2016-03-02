# This is  the code for extracting Test Loss information from output For Classification

def get_loss(lines):
    for idx in range(len(lines)):
        line = lines[idx]
        if 'testMSE' in line and 'Train' not in line:
            loss = float(line.split('Loss = ')[1].strip()[:-1])
            return loss, idx

if __name__ == '__main__':
    with open('output') as f:
        lines = f.readlines()

    test_label_lines = []
    for idx in range(len(lines)):
        if 'Test @' in lines[idx]:
            test_label_lines.append(idx)
    test_label_lines.append(len(lines))

    case_lines = []
    for idx in range(1, len(test_label_lines)):
        case_lines.append(lines[test_label_lines[idx - 1]:test_label_lines[idx]])

    min_loss = 1e100
    min_idx = 0
    min_loc = 0
    for idx in range(len(case_lines)):
        t_loss, t_loc = get_loss(case_lines[idx])
        if t_loss < min_loss:
            min_loss = t_loss
            min_loc = t_loc
            min_idx = idx

    truth = []
    pred = []
    target_lines = case_lines[min_idx]
    for idx in range(min_loc):
        if '******Loss Layer****** Ground truth value, Estimate value:' in target_lines[idx]:
            value_str = target_lines[idx].split('******Loss Layer****** Ground truth value, Estimate value:')[1]
            values = [i.strip() for i in value_str.split(' ') if i.strip()]
            truth.append(values[0])
            pred.append(values[1])

    with open('extracted_truth_pred.txt', 'w') as f:
        for t, p in zip(truth, pred):
            f.write('{},{}\n'.format(t, p))