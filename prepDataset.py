import pandas as pd


def map_added_removed(added_removed):
    adf, rmf = added_removed
    objs = []
    for i in range(len(adf)):
        objs.append({'added_code': [l.encode('utf8', 'surrogateescape').decode('utf8', 'replace') for l in adf[i]],
                     'removed_code': [l.encode('utf8', 'surrogateescape').decode('utf8', 'replace') for l in rmf[i]]})
        #if i == 0:
         #   print(type(objs[0]['added_code']))
    return objs


def seperate_added_removed(added_code, removed_code):
    return list(map(map_added_removed, zip(added_code, removed_code)))


added_token, removed_token = '<added>', '<removed>'


def add_added_removed_tokens(adf, rmf):
    # adf,rmf = c_codes
    af_len = len(adf)
    rf_len = len(rmf)
    res = [] * max(af_len, rf_len)
    for i in range(min(af_len, rf_len)):
        # remove surrogate characters in each line and insert added/removed tokens
        adf[i] = [added_token + ' ' + l.encode('utf8', 'surrogateescape').decode('utf8', 'replace') for l in adf[i]]
        rmf[i] = [removed_token + ' ' + l.encode('utf8', 'surrogateescape').decode('utf8', 'replace') for l in rmf[i]]
        res.append(adf[i] + rmf[i])
    if af_len > rf_len:
        for j in range(af_len):
            adf[j] = [added_token + ' ' + l.encode('utf8', 'surrogateescape').decode('utf8', 'replace') for l in adf[j]]
            res.append(adf[j])
    elif rf_len > af_len:
        for j in range(af_len):
            rmf[j] = [added_token + ' ' + l.encode('utf8', 'surrogateescape').decode('utf8', 'replace') for l in rmf[j]]
            res.append(rmf[j])
    return res  # + rmf


def group_added_removed(added_code, removed_code):
    # return list(map(add_added_removed_tokens, added_code, removed_code))
    def add_added_token(lines):
        return list(map(lambda l: '<added> ' + l, lines))

    def add_removed_token(lines):
        return list(map(lambda l: '<removed> ' + l, lines))

    def map_files(ac, rc):
        added = list(map(add_added_token, ac))
        removed = list(map(add_removed_token, rc))
        return list(map(lambda a, b: a + b, added, removed))

    if multiple_files:
        return list(map(map_files, added_code, removed_code))
    print('single files')
    added = list(map(add_added_token, added_code))
    removed = list(map(add_removed_token, removed_code))
    return list(map(lambda a, b: a + b, added, removed))


def transformDataFrame(df, codes_format='two_blocks'):
    new_df = pd.DataFrame()
    new_df[0] = df['commit_id']
    new_df[1] = df['buggy'].astype(int)
    if type(df['msg'].iloc[0]) == list:
        # Convert messages from list of words to string.
        new_df[2] = df['msg'].str.join(' ')
    else:
        new_df[2] = df['msg']

    if codes_format == 'one_block':
        new_df[3] = group_added_removed(df['added_code'], df['removed_code'])
    elif codes_format == 'two_blocks':
        new_df[3] = seperate_added_removed(df['added_code'], df['removed_code'])
    else:
        new_df[3] = df['added_code'] + df['removed_code']
    return new_df


### Adjust test ids
# Turns out the test_ids originally selected (from the given apachejit dataset) are imbalanced for each project.
# Here I will select new test ids based on a fixed percent.
# Since all datasets are ordered by the commit year, I simply take percentage of commits required from the end.
def getTestIds(test_percent, df):
    return df[-int(df.shape[0] * test_percent):]['commit_id']


def splitTrainAndTest(p, test_percent):
    # p.table = transformDataFrame(p.total)
    commits_set = set(getTestIds(test_percent, p.total))
    test_idxs = [i for i, c in enumerate(p.table[0]) if c in commits_set]
    train_idxs = [i for i in range(len(p.table[0])) if i not in test_idxs]
    train = p.table.loc[train_idxs]
    test = p.table.loc[test_idxs]

    p.train = {'id': train[0].tolist(), 'label': train[1].tolist(), 'msg': train[2].tolist(), 'code': train[3].tolist()}
    p.test = {'id': test[0].tolist(), 'label': test[1].tolist(), 'msg': test[2].tolist(), 'code': test[3].tolist()}
    p.table = {'id': p.table[0].tolist(), 'label': p.table[1].tolist(), 'msg': p.table[2].tolist(),
               'code': p.table[3].tolist()}
    # print('Train {}:{} Test {}:{}'.format(p.total.shape[0] - len(p.test_ids),len(p.cc2vecTrain),len(p.test_ids),len(p.cc2vecTest)))


def split_train_valid_test(p, valid_percent, test_percent):
    test_len = int(p.table.shape[0] * test_percent)
    valid_len = int(p.table.shape[0] * valid_percent)
    train_len = int(p.table.shape[0] * (1 - test_percent - valid_percent))

    test = p.table[train_len + valid_len:]
    valid = p.table[train_len:(train_len + valid_len)]
    train = p.table[:train_len]
    #print('{} {} {}'.format(train.shape[0], valid.shape[0], test.shape[0]))
    p.train = {'id': train[0].tolist(), 'label': train[1].tolist(), 'msg': train[2].tolist(), 'code': train[3].tolist()}
    p.valid = {'id': valid[0].tolist(), 'label': valid[1].tolist(), 'msg': valid[2].tolist(), 'code': valid[3].tolist()}
    p.test = {'id': test[0].tolist(), 'label': test[1].tolist(), 'msg': test[2].tolist(), 'code': test[3].tolist()}
    p.table = {'id': p.table[0].tolist(), 'label': p.table[1].tolist(), 'msg': p.table[2].tolist(),
               'code': p.table[3].tolist()}


use_lowercase = True


def add_dict(dict, key):
    if use_lowercase:
        key = key.lower()
    if key in dict:
        dict[key] += 1
    else:
        dict[key] = 1


def set_in_dict(dict, key, i):
    if use_lowercase:
        key = key.lower()
    if not key in dict:
        dict[key] = i
        return i + 1
    return i


def add_msg_words(d, msg):
    for w in msg.split(' '):
        add_dict(d, w)
        # ctr = set_in_dict(dict,w,ctr) unique_msg_words = ctr


### Adds all words found in codes to the dictionary. Assumes each element in the list is a file.
#  And each file is an object with two lists, one for added code and one for removed code.
def add_code_words(d, c_codes):
    for adrm in c_codes:
        for al in adrm['added_code']:
            for aw in al.split(' '):
                add_dict(d, aw)
                # unique_code_words = set_in_dict(dict,aw,unique_code_words)
        for rl in adrm['removed_code']:
            for rw in rl.split(' '):
                add_dict(d, rw)
                # unique_code_words = set_in_dict(dict,rw,unique_code_words)


### Adds all words found in codes to the dictionary. Assumes each element in the list is a file.
#  And each file is a list of lines for both added and removed code grouped together.
def add_grouped_code_words(dict, c_codes):
    for f in c_codes:
        for al in f:
            for aw in al.split(' '):
                add_dict(dict, aw)


def createDictionaryFromTable(table, code_format='one_block',min_frequency=3):
    # unique_msg_words = 1
    # unique_code_words = 1
    #print(table)
    msg_dict, code_dict = dict(), dict()
    msg_dict['<NULL>'] = 0
    code_dict['<NULL>'] = 0
    table[2].apply(lambda m: add_msg_words(msg_dict, m))
    if code_format == 'one_block':
        table[3].apply(lambda c: add_grouped_code_words(code_dict, c))
    elif code_format == 'two_blocks':
        table[3].apply(lambda c: add_code_words(code_dict, c))
    # order dict by frequency and remove words that occur less than min_frequency times
    msg_dict = {k:v for k,v in sorted(msg_dict.items(),key = lambda item: item[1],reverse=True) if v >= min_frequency}
    code_dict = {k:v for k,v in sorted(code_dict.items(),key = lambda item: item[1],reverse=True) if v >= min_frequency}
    # create new dictionary with only the words that occur more than min_frequency times
    msg_dict = {k:i for i,k in enumerate(msg_dict.keys(),1)}
    code_dict = {k:i for i,k in enumerate(code_dict.keys(),1)}
    # add null token
    msg_dict['<NULL>'] = 0
    code_dict['<NULL>'] = 0
    msg_dict = {key: index for index, key in enumerate(msg_dict,1)}
    code_dict = {key: index for index, key in enumerate(code_dict,1)}
    return (msg_dict, code_dict)


### Prepare project data for training by models
# test_percent: percentage of commits to take as test instances.
# added_removed_format: 'one_block' or 'two_blocks'.
# one_block groups all added and removed code together, by prepending a token before each added and remove line.
# two_blocks keeps all added lines and removed lines seperate.
def prepareDataset(p, test_percent=0, valid_percent=0, added_removed_format='one_block', tokenize=False,
                   lowercase_tokens=True,
                   id_col='commit_id', added_code_col='added_code', removed_code_col='removed_code', msg_col='msg',
                   label_col='buggy'):
    global use_lowercase
    use_lowercase = lowercase_tokens
    global multiple_files
    multiple_files = p.multiple_files
    # rename total dataframe based on given arguments
    p.total = p.total.rename(
        columns={id_col: 'commit_id', added_code_col: 'added_code', removed_code_col: 'removed_code', msg_col: 'msg',
                 label_col: 'buggy'})
    p.table = transformDataFrame(p.total, added_removed_format)
    p.dict = createDictionaryFromTable(p.table,code_format=added_removed_format)
    if tokenize:
        tokenizeTable(p.table, p.dict)
    if valid_percent > 0:
        split_train_valid_test(p, valid_percent, test_percent)
    elif test_percent > 0:
        splitTrainAndTest(p, test_percent)
    # rename cols back
    p.total = p.total.rename(
        columns={'commit_id': id_col, 'added_code': added_code_col, 'removed_code': removed_code_col, 'msg': msg_col,
                 'buggy': label_col})
    return p


def convertCodeFormat(p, code_column='code', from_type='one_block', to='two_blocks'):
    if p.total is None:
        print("No data loaded")
        return
    # check if column exists
    if not code_column in p.total.columns:
        print("Column not found")
        return
    # check if column is in the right format
    if from_type == 'one_block':
        codes = p.total[code_column].iloc[0]
        if type(codes) != list:
            codes = eval(codes)
        if type(codes) != list:
            print("Column is not in the right format")
            return
        if p.multiple_files:
            if type(codes[0]) != list:
                print("Column is not in the right format")
                return
        # convert each line of <added> ... or <removed> ... to a list of dictionaries. Each dictionary contains an entry for added lines and an entry for removed lines.
        if p.multiple_files:
            p.total[code_column] = p.total[code_column].apply(lambda c: \
                                                                  [{'added_code': [line for line in f if
                                                                                   line.startswith('<added>')],
                                                                    'removed_code': [line for line in f if
                                                                                     line.startswith('<removed>')]} for
                                                                   f in c])
        else:
            p.total[code_column] = p.total[code_column].apply(lambda c: \
                                                                  {'added_code': [line for line in c if
                                                                                  line.startswith('<added>')],
                                                                   'removed_code': [line for line in c if
                                                                                    line.startswith('<removed>')]})
    elif from_type == 'two_blocks':
        codes = p.total[code_column].iloc[0]
        if type(codes) != list:
            codes = eval(codes)
        if type(codes) != list:
            print("Column is not in the right format")
            return
        if type(codes[0]) != dict:
            print("Column is not in the right format")
            return
        # Convert each list of dictionaries to a list of lines. Each line is a string of <added> ... or <removed> ...
        if p.multiple_files:
            p.total[code_column] = p.total[code_column].apply(lambda c: \
                                                                  [['<added> ' + line for line in f['added_code']] +
                                                                   ['<removed> ' + line for line in f['removed_code']]
                                                                   for f in c])
        else:
            p.total[code_column] = p.total[code_column].apply(lambda c: \
                                                                  ['<added> ' + line for line in c['added_code']] + \
                                                                  ['<removed> ' + line for line in c['removed_code']])

    else:
        print("Unknown format")
        return


def correctCase(w):
    if use_lowercase:
        return w.lower()
    return w


def tokenizeTable(table, dict):
    def tokenizeMsg(msg, msg_dict):
        t_msg = [msg_dict[correctCase(w)] for w in msg.split(' ')]
        return t_msg

    def tokenizeCodeChange(code, c_dict):
        def tokenizeFile(file):
            if type(file) == list:
                return [[c_dict[correctCase(w)] for w in line.split(' ')] for line in file]
            else:
                return {'added_code': [[c_dict[correctCase(w)] for w in line] for line in file['added_code']],
                        'removed_code': [[c_dict[correctCase(w)] for w in line] for line in file['removed_code']]}

        return [tokenizeFile(f) for f in code]

    table[2] = table[2].apply(lambda m: tokenizeMsg(m, dict[0]))
    table[3] = table[3].apply(lambda c: tokenizeCodeChange(c, dict[1]))
    return table
