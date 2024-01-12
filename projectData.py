from cmath import nan
import re
import pandas as pd
import concurrent
from concurrent import futures

comments_start = ('//', '*', 'F/*', '<!--', '#')
block_comments = ('/*', '<!--')
comments_end = ('*/', '-->')
symbolsToRemove = ['{', '}', ';', '\r']

# if in one line, we find statements after one of those symbols, we add a new line right after each occurence of these symbols
lines_end_split = [';', '{']
# if a line ends with one of those symbols, we group the next line with it (remove newline character); only for changed lines
lines_end_group = [',']
ignored_files = ['txt']
processed_files = ['java']


def removeEmpty(strList):
    return list(filter(None, strList))


def removeSymbols(line):
    line = re.sub('[{};]+', '', line)
    return line


def createSymbolsRegex(symbols, group=True):
    s = '([' if group else '['
    for sm in symbols:
        s += f'\{sm}'
    s += '])' if group else ']'
    return s


# regex replace functions
def numrepl(matchobj):
    return matchobj.group(1) + '<NUM>'


def newline_rpl(matchobj):
    return '\n' + matchobj.group(1) + matchobj.group(2)


def group_end_lines(file):
    def replace(matchobj):
        return matchobj.group(1) + matchobj.group(2)

    regex = '^(?P<c>[\+\-])([^\+\-\n].*?,)(\n(?P=c)[\t ]*)'
    n = 1
    while n > 0:
        (file, n) = re.subn(regex, replace, file, 1)
    regex = '^((?P<c>[\+\-]).*\))\n((?P=c)[\t ]*){(.*)'
    n = 1
    while n > 0:
        (file, n) = re.subn(regex, '\\g<1>{\n\\g<3>\\g<4>', file, 1)
    return file


def split_end_lines(file):
    regex = '^((\+|\-).*' + createSymbolsRegex(lines_end_split, False) + ')([\t ]*[^\s]+)'
    n = 1
    subst = "\\g<1>\\n\\g<2>\\g<3>"
    while n > 0:
        (file, n) = re.subn(regex, subst, file, 1, re.MULTILINE)
    return file


def extractAddedRemoveFromDiff(source):
    added_lines = removeEmpty(re.findall("^\+(?:[\t ]+|\+*)([^\+\r\n\t\f\v ].*)", source, re.MULTILINE))
    removed_lines = removeEmpty(re.findall("^\-(?:[\t ]+|\-*)([^\-\r\n\t\f\v ].*)", source, re.MULTILINE))
    return added_lines, removed_lines


def java_comments_repl(matchobj):
    if matchobj.group(1):
        return matchobj.group(1)
    return ''


class ProjectData:
    def __init__(self, name, df, lang='Java', train_ids=None, test_ids=None, multiple_files=True):
        self.name = name
        self.total = df.copy()
        self.train_ids = train_ids
        self.test_ids = test_ids
        self.multiple_files = multiple_files
        self.lang = lang
        if lang == 'Java' or lang == 'C' or lang == 'C++':
            self.comments_regex = '\/\*[\S\s]*?\*\/|\/\*[\S\s]*|\/\/.*|^\*.*\n?|(".*")'  # '(\/\*.*)|^(?:\s|[+-])*(\*.*\n?)|(.*\*\/\n?)|(\/\/.*)|(<!--[\S\s]*-->)'
            self.comments_repl = java_comments_repl
        elif lang == 'Python':
            self.comments_regex = '(#.*)|("""[\S\s]*?""")|(\'\'\'[\S\s]*?\'\'\')|("""[\S\s]*)'
            self.comments_repl = ''
        self.symbols_regex = createSymbolsRegex(symbolsToRemove)
        # self.comments_regex += '|' + createSymbolsRegex(symbolsToRemove) +'|'

    def setup(self):
        # Java, C++ and Python comments.
        # self.comments_regex = '(\/\*.*)|^(?:\s|[+-])*(\*.*\n?)|(.*\*\/\n?)|(\/\/.*)|(<!--[\S\s]*-->)|("""[\S\s]*?""")|(#.*)'
        if self.lang == 'Java' or self.lang == 'C' or self.lang == 'C++':
            self.comments_regex = '\/\*[\S\s]*?\*\/|\/\*[\S\s]*|\/\/.*|(".*")'  # '//.*|/\*(?s:.*?)\*/|("(?:(?<!\\)(?:\\\\)*\\"|[^\r\n"])*")'
            # '(\/\*[\S\s]*?\*\/)|(\/\*[\S\s]*)|([\S\s]*?\*\/)|(\/\/.*)' #'(\/\*.*)|^(?:\s|[+-])*(\*.*\n?)|(.*\*\/\n?)|(\/\/.*)|(<!--[\S\s]*-->)'
        elif self.lang == 'Python':
            self.comments_regex = '(#.*)|("""[\S\s]*?""")|(\'\'\'[\S\s]*?\'\'\')|("""[\S\s]*)'
        self.symbols_regex = createSymbolsRegex(symbolsToRemove)

    def preProcess(self, f, remove_comments=False):
        if remove_comments:
            f = re.sub(self.comments_regex, self.comments_repl, f, 0, re.MULTILINE)  # 0 for infinite.
        f = re.sub('".+"', '<STR>', f)  # double quoted python strings
        # f = re.sub("'.+'", '<STR>', f) # single quoted python strings
        # remove \n and \r characters
        f = re.sub(r'\\[rn]', '', f)
        # f = group_end_lines(f)  # group parameters of calls to functions (if all is added or removed)
        # f = re.sub('^(?:[^\+\-]|[\+\-]{2,}).*\n', '', f)  # filter for changed lines only

        # f = split_end_lines(f)  # split two statements in one line
        ### Note: Do not remove symbols '{};' now because we need them to distinguish between nodes types 
        #         (like constructors and methods) later on.
        # f = re.sub(self.symbols_regex,'',f) 
        f = re.sub('([^a-zA-Z0-9_]){1}\d+(?:\.\d+)?f?', numrepl, f)
        return f

    def preProcessFiles(self, files, remove_comments=False):
        return list(map(lambda f: removeEmpty(list(map(lambda l: self.preProcess(l, remove_comments), f))), files))

    def eval_to_list(self, col_name):
        if len(self.total[col_name]) > 0 and type(self.total[col_name].iloc[0]) == str:
            self.total[col_name] = list(map(lambda i: eval(i), self.total[col_name]))
        else:
            print('Asked to eval str into list, but {} column already has type: {}'.format(col_name, type(
                self.total[col_name].iloc[0])))

    def doPreProcessingOnDataset(self, added_code_col='added_code', removed_code_col='removed_code', code_col='code',
                                 separate_add_rem=True, remove_comments=False):
        if separate_add_rem:
            if type(self.total[added_code_col].iloc[0]) == str:
                self.eval_to_list(added_code_col)
            if type(self.total[removed_code_col].iloc[0]) == str:
                self.eval_to_list(removed_code_col)
            if self.multiple_files:
                self.total[added_code_col] = self.total[added_code_col].map(
                    lambda n: self.preProcessFiles(n, remove_comments))
                self.total[removed_code_col] = self.total[removed_code_col].map(
                    lambda n: self.preProcessFiles(n, remove_comments))
            else:
                self.total[added_code_col] = self.total[added_code_col].map(
                    lambda n: removeEmpty(list(map(lambda l: self.preProcess(l, remove_comments), n))))
                self.total[removed_code_col] = self.total[removed_code_col].map(
                    lambda n: removeEmpty(list(map(lambda l: self.preProcess(l, remove_comments), n))))
        else:
            if type(self.total[code_col].iloc[0]) == str:
                self.eval_to_list(code_col)
            if self.multiple_files:
                self.total[code_col] = self.total[code_col].map(lambda n: self.preProcessFiles(n, remove_comments))
            else:
                self.total[code_col] = self.total[code_col].map(
                    lambda n: removeEmpty(list(map(lambda l: self.preProcess(l, remove_comments), n))))

    def extractAddedRemoveFromDiff(self, source, keep_unchanged_lines=True, separate_added_removed=True):
        def remove_comments(b):
            b = re.sub(self.comments_regex, self.comments_repl, b, 0, re.MULTILINE)
            return b

        # split diff into blocks based on the line starting with @@.
        blocks = re.split('(?=^@@)', source, 0, flags=re.MULTILINE)
        blocks = blocks[1:]
        # remove comments each block and merge
        changes = ''.join(list(map(lambda b: remove_comments(b), blocks)))
        # remove all lines that start with  @@ or +++ or ---, --git, or index
        changes = re.sub('^ *(?:@@|\+{3}|-{2}|index).*', '', changes, 0, re.MULTILINE)
        #  remove empty lines even if start with + or - #"^\+(?:[\t ]+|\+*)([^\+\r\n\t\f\v ].*)"
        changes = re.sub('^(?:\+|\-)?[\t\f\v ]*(\n|\r)', '', changes, 0, re.MULTILINE)

        if not separate_added_removed:
            if keep_unchanged_lines:
                # for lines that do not start with + or - add <keep> in beginning only if there are characters in the line
                changes = re.sub('^(?![+\-])[\t\v ]*(.+)$', '<keep> \\1', changes, 0, re.MULTILINE)
            else:
                # remove lines that do not start with + or -:
                changes = re.sub('^(?![+\-]).*\n', '', changes, 0, re.MULTILINE)
            # change + in beginning to <add> only if there are characters in the line
            changes = re.sub('^\+[\t\v ]*(.+)$', '<add> \\1', changes, 0, re.MULTILINE)
            # change - in beginning to <del> only if there are characters in the line
            changes = re.sub('^\-[\t\v ]*(.+)$', '<del> \\1', changes, 0, re.MULTILINE)
            return changes.split('\n')
        else:
            added_lines = removeEmpty(re.findall("^\+(?:[\t ]+|\+*)([^\+\r\n\t\f\v ].*)", changes, re.MULTILINE))
            removed_lines = removeEmpty(re.findall("^\-(?:[\t ]+|\-*)([^\-\r\n\t\f\v ].*)", changes, re.MULTILINE))
            return added_lines, removed_lines

    def getAddedRemovedFromFiles(self, files, filenames, obj):
        files_added_code = [''] * len(files)
        files_removed_code = [''] * len(files)
        i = 0
        for f in files:
            if filenames[i] in processed_files:
                # print(f)
                # added_lines = removeEmpty(re.findall("^\+(?:[\t ]+|\+*)([^\+\r\n\t\f\v ].*)", f, re.MULTILINE))
                # removed_lines = removeEmpty(re.findall("^\-(?:[\t ]+|\-*)([^\-\r\n\t\f\v ].*)", f, re.MULTILINE))
                f = self.preProcess(f)
                added_lines, removed_lines = self.extractAddedRemoveFromDiff(f)
            else:
                added_lines = []
                removed_lines = []
            files_added_code[i] = added_lines
            files_removed_code[i] = removed_lines
            i += 1
            # print('Authored date: {} Datatime: {} Author Date: {}'.format(commit.authored_date,commit.authored_datetime,commit.committed_date))
        obj['files'] = files
        obj['added'] = files_added_code
        obj['removed'] = files_removed_code
        obj['files_exts'] = filenames
        return obj

    def getChangeStringFromFiles(self, files, filenames, obj):
        # get the change string by preprocessing files but keeping the added and removed lines in one string. Also
        # keep unchanged line. Keep order of lines.
        files_code = [''] * len(files)
        i = 0
        for f in files:
            if filenames[i] in processed_files:
                # print(f)
                # added_lines = removeEmpty(re.findall("^\+(?:[\t ]+|\+*)([^\+\r\n\t\f\v ].*)", f, re.MULTILINE))
                # removed_lines = removeEmpty(re.findall("^\-(?:[\t ]+|\-*)([^\-\r\n\t\f\v ].*)", f, re.MULTILINE))
                f = self.preProcess(f)
                changes = self.extractAddedRemoveFromDiff(f, separate_added_removed=False)
            else:
                changes = []
            files_code[i] = changes
            i += 1
            # print('Authored date: {} Datatime: {} Author Date: {}'.format(commit.authored_date,
            # commit.authored_datetime,commit.committed_date))
        obj['files'] = files
        obj['code'] = files_code
        obj['files_exts'] = filenames
        return obj

    def extractAddedRemovedFromDF(self, code_column='code', added_code_col='added_code',removed_code_col='removed_code'):
        def extractFromFiles(files):
            files_added_code, files_removed_code = [], []
            for f in files:
                added_code, removed_code = self.extractAddedRemoveFromDiff(f)
                files_added_code.append(added_code)
                files_removed_code.append(removed_code)
            return files_added_code, files_removed_code

        if self.multiple_files:
            self.total[added_code_col], self.total[removed_code_col] = zip(
                *list(map(extractFromFiles, self.total[code_column])))
        else:
            self.total[added_code_col], self.total[removed_code_col] = zip(
                *list(map(self.extractAddedRemoveFromDiff, self.total[code_column])))

    def extractChangesFromDF(self, code_column='code',keep_unchanged_lines=True):
        def extractFromFiles(files):
            files_code = []
            for f in files:
                code = self.extractAddedRemoveFromDiff(f, separate_added_removed=False,keep_unchanged_lines=keep_unchanged_lines)
                files_code.append(code)
            return files_code

        if self.multiple_files:
            self.total[code_column] = list(map(extractFromFiles, self.total[code_column]))
        else:
            self.total[code_column] = list(map(lambda c: self.extractAddedRemoveFromDiff(c, separate_added_removed=False,keep_unchanged_lines=keep_unchanged_lines),self.total[code_column]))

    def extractLinesFromDF(self, preprocess=True, code_column="code", separate_add_rem=True, keep_unchanged_lines=True,added_code_col='added_code', removed_code_col='removed_code'):
        if separate_add_rem:
            self.extractAddedRemovedFromDF(code_column, added_code_col, removed_code_col)
        else:
            self.extractChangesFromDF(code_column,keep_unchanged_lines=keep_unchanged_lines)
        if preprocess:
            self.doPreProcessingOnDataset(code_col=code_column,separate_add_rem=separate_add_rem)
        return self.total

    def extractCommitLines(self, repo, c, separate_added_removed=True):
        try:
            #print(c)
            commit = repo.commit(c)
        except ValueError:
            return {'message': '', 'files': '', 'files_exts': '', 'added': [], 'removed': []}
        if len(commit.parents) == 0:
            try:
                diff = repo.git.show(c)
                # diff = commit.tree.diff_to_tree()
                # stats = diff.stats
                # diff= diff.patch
            except ValueError:
                return {'message': '', 'files': '', 'files_exts': '', 'added': [], 'removed': []}
        else:
            p_commit = commit.parents[0]
            diff = repo.git.diff(p_commit, commit)
            # diff = repo.diff(pCommit, commit)
            # stats = diff.stats
            # diff = diff.patch
        files = re.split("^diff", diff, flags=re.M)

        files = files[1:]
        files_added_code = [''] * len(files)
        files_removed_code = [''] * len(files)
        i = 0

        # filenames = [e.name for e in commit.tree]
        # filenames = list(map(lambda name: name.split('.')[-1], commit.stats.files.keys()))
        # obtain filenames from first line of each file using regex: --git path/to/file.ext
        # In cases of file renames, using commit.tree or diff.stats to get the filenames give wrong results.
        def get_file_name(f):
            res = re.search(r'--git.*\/([\w-]+)?\.?([\w-]+)', f)
            if res:
                # check if there is a second group (extension)
                if res.group(2):
                    return res.group(2)
                else:
                    return res.group(1)
            else:
                print(f)
                return ''

        filenames = list(map(get_file_name, files))
        # just in case, check for this:
        if len(filenames) != len(files):
            print(len(commit.parents))
            print('Filenames: {} Files: {}'.format(len(filenames), len(files)))
            print(filenames)
            print(c)
            print(commit.stats.files)

        obj = {'message': commit.message}
        if separate_added_removed:
            obj = self.getAddedRemovedFromFiles(files, filenames, obj)
        else:
            obj = self.getChangeStringFromFiles(files, filenames, obj)

        return obj

    def extractCommitAndUpdateDf(self, repo, gitt, c, idx):
        obj = self.extractCommitLines(repo, gitt, c)
        self.total.at[idx, 'commit_message'] = obj['message']
        self.total.at[idx, 'added_code'] = obj['added']
        self.total.at[idx, 'removed_code'] = obj['removed']
        self.total.at[idx, 'files_exts'] = obj['files_exts']

    def extractCommits(self, project_repo):
        # gitt =
        self.total['commit_message'] = ""
        self.total['added_code'] = ""
        self.total['removed_code'] = ""
        self.total['files_exts'] = ""
        idx = 0
        all_patches = []
        for i, c in enumerate(self.total['commit_id']):
            #print(c)
            all_patches.append(self.extractCommitLines(project_repo, c))
            print('Extracted commit {} of {}'.format(i, len(self.total['commit_id'])))
        # with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        #   ctr = 0
        #  futures = {executor.submit(self.extractCommitAndUpdateDf,project_repo,gitt,c,idx) for idx, c in enumerate(self.total['commit_id'])}
        print('Done extracting commits from repo (added & removed code, message, files)')
        self.total['added_code'] = pd.Series(map(lambda d: d['added'], all_patches))
        self.total['removed_code'] = pd.Series(map(lambda d: d['removed'], all_patches))
        self.total['commit_message'] = pd.Series(map(lambda d: d['message'], all_patches))
        self.total['files_exts'] = pd.Series(map(lambda d: d['files_exts'], all_patches))
        self.total['files'] = pd.Series(map(lambda d: d['files'], all_patches))
        print('index: {} all_patches {} dataframe {}'.format(idx, len(all_patches), self.total.shape))
        self.cleanNulls()

    def cleanNulls(self):
        nans = self.total.isna()
        print(
            'Added & removed nan: {}'.format(nans[(nans['added_code'] == True) & (nans['removed_code'] == True)].shape))
        self.total.dropna(axis=0, subset=['added_code', 'removed_code'], how='all', inplace=True)
        self.total['added_code'] = self.total['added_code'].fillna('')
        self.total['removed_code'] = self.total['removed_code'].fillna('')
        nans = self.total.isna()
        print('{} Added & removed nan: {}'.format(self.total.shape, nans[
            (nans['added_code'] == True) & (nans['removed_code'] == True)].shape))

# %%
