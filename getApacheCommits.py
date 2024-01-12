import pandas as pd
import re
import git
import json
from git import Repo
from projectData import ProjectData
import os
import pickle

debug = False
updateTotalFile = False
current_directory = os.getcwd()


def GetApacheDataSets():
    total = pd.read_csv(current_directory + '/apachejit/dataset/apachejit_total.csv')
    train = pd.read_csv(current_directory + '/apachejit/dataset/apachejit_train.csv')
    test = pd.read_csv(current_directory + '/apachejit/dataset/apachejit_test_large.csv')
    return [total, train, test]


def ExtractProjectOnly(df, name):
    df_a = df[df['project'] == name].copy().reset_index(drop=True)
    df_a['project'] = df_a['project'].str.split(pat="/", expand=True)[1]
    return df_a


def RemoveEmpty(strList):
    return list(filter(None, strList))


apache_total, apache_train, apache_test = GetApacheDataSets()
projects_arr = apache_total['project'].unique()
projects = [0] * len(projects_arr)
projects_map = {}
i = 0
for a in projects_arr:
    projects_map[a.split('/')[1]] = i
    total = ExtractProjectOnly(apache_total, a)
    train_ids = ExtractProjectOnly(apache_train, a)['commit_id']
    test_ids = ExtractProjectOnly(apache_test, a)['commit_id']
    projects[i] = ProjectData(a.split('/')[1], df = total, train_ids=train_ids, test_ids=test_ids, lang='Java')
    i += 1

if updateTotalFile and os.path.exists(current_directory + '/totalApacheCommits.pkl'):
    with open(current_directory + '/totalApacheCommits.pkl', 'rb') as f:
        total_projects_df = pickle.load(f)
else:
    total_projects_df = pd.DataFrame().reindex_like(apache_total)
    total_projects_df = total_projects_df[0:0]
if not os.path.exists(current_directory + '/apache_sources'):
    os.mkdir(current_directory + '/apache_sources')
for p in projects:
    print('project: {} dataframe: {}'.format(p.name, p.total.shape))
    project_repo = ''
    if os.path.exists(current_directory + "/apache sources/{}/".format(p.name)):
        project_repo = Repo(current_directory + "/apache sources/{}/".format(p.name))
    else:
        project_repo = Repo.clone_from("https://github.com/apache/{}".format(p.name),
                                       current_directory + "/apache sources/{}/".format(p.name))
    p.total['commit_message'] = ""
    all_patches = []

    p.extractCommits(project_repo)

    with open(current_directory + '/apache_sources/{}_fulldata.pkl'.format(p.name), 'wb') as f:
        pickle.dump(p.total, f)
    total_projects_df = pd.concat([total_projects_df, p.total], axis=0, ignore_index=True)
total_projects = ProjectData("Total", total_projects_df)

if debug:
    print('### Before pickling: ')
    print(total_projects_df.iloc[28]['removed_code'])

with open(current_directory + '/apache_sources/totalApacheCommits.pkl', 'wb') as f:
    pickle.dump(total_projects_df, f)
if debug:
    with open(current_directory + '/apache_sources/totalApacheCommits.pkl', 'rb') as f:
        total_projects_df = pickle.load(f)

    print('### After pickling: ')
    print(total_projects_df.iloc[28]['removed_code'])