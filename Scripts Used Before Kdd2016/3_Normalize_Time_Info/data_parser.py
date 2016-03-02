def parsing(idx, text, parse_item, spliter, title):
    if parse_item:
        split_idx = parse_item[0]
        split_cha = spliter[0]
        segs = text.split(split_cha)
        iters = range(len(segs)) if split_idx == -1 else [split_idx]
        for it in iters:
            parsing(idx, segs[it], parse_item[1:], spliter[1:], title + [str(it)])
    else:
        titles.add('-'.join(title))
        result.setdefault(idx, dict())['-'.join(title)] = text


if __name__ == '__main__':
    split_char = ['\001', '\002', '\003', '\004']

    parse_items = [
        [0, 0, 0],
        [0, -1, 1],
        [1, 1]
    ]

    #infile_path = 'train.shard'
    infile_path = 'test.shard'
    #infile_path = 'valid.shard'
    #oufile_path = 'parsing_result_train.csv'
    oufile_path = 'parsing_result_test.csv'
    #oufile_path = 'parsing_result_valid.csv'

    with open(infile_path) as f:
        lines = [line.strip() for line in f.readlines()]

    result = dict()
    titles = set()

    for idx in range(len(lines)):
        for item in parse_items:
            parsing(idx, lines[idx], item, split_char, [])

    titles = sorted(list(titles))

    with open(oufile_path, 'w') as f:
        f.write('Total lines: {}\n'.format(len(lines)))
        f.write(','.join(titles) + '\n')
        for idx in range(len(lines)):
            f.write(','.join([result[idx].setdefault(title, '') for title in titles]) + '\n')

