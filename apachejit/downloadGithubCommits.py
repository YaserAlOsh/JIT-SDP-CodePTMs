
query = """
query ($name: String!, $owner: String!, $branch: String!){
  repository(name: $name, owner: $owner) {
    ref(qualifiedName: $branch) {
      target {
        ... on Commit {
          history(first: 1, after: %s) {
            nodes {
              message
              committedDate
              authoredDate
              oid
              author {
                email
                name
              }
            }
            totalCount
            pageInfo {
              endCursor
            }
          }
        }
      }
    }
  }
}
"""

def getHistory(cursor):
    r = requests.post("https://api.github.com/graphql",
        headers = {
            "Authorization": f"Bearer {token}"
        },
        json = {
            "query": query % cursor,
            "variables": {
                "name": name,
                "owner": owner,
                "branch": branch
            }
        })
    return r.json()["data"]["repository"]["ref"]["target"]["history"]


import requests
import os
from pprint import pprint

token = "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"
owner = "Apache"
repo = "groovy"
commit_id = "3a0e164308dbf76322cc7473f6181694ca7e4867"
query_url = f"https://api.github.com/repos/{owner}/{repo}/commits/{commit_id}"
params = {}
headers= {'Authorization':f'token{token}'}
r = requests.get(query_url,headers=headers,params=params)
#pprint(r.json())
patch = r.json()['files'][0]['patch']

#pprint(patch.split("\n",2)[2])
print(patch.split("\n",2)[2])
print(len(r.json()['files']))
pprint(type(patch))