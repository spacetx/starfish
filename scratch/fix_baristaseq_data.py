# what follows is the history that was needed to fix the primaryFOV_000.json file
import json
dat = json.load('primaryFOV_000.json')
with open('primaryFOV_000.json') as f:
    dat = json.load(f)
with open('primaryFOV_000.json') as f:
    dat = f.read(150)
dat
pprint(dat)
pprint(dat)
from pprint import pprint
pprint(dat)
with open('primaryFOV_000.json') as f:
    dat = json.load(f)
import re

def clean_json(string):
    string = re.sub(",[ \t\r\n]+}", "}", string)
    string = re.sub(",[ \t\r\n]+\]", "]", string)

    return string
with open('primaryFOV_000.json') as f:
    dat = f.read()
cleaned = clean_json(dat)
json = json.loads(cleaned)
jon
jon
json
json_dat = json
import json
json.dump(json_dat, "primary_corrected_FOV_000.json")
with open("primary_corrected_FOV_000.json") as f:
    json.dump(json_dat, f)
with open("primary_corrected_FOV_000.json", 'w') as f:
    json.dump(json_dat, f)
json_dat
json_dat.keys()
json_dat['tiles'].keys()
json_dat['tiles'][0].keys()
for t in json_dat['tiles']:
    for i in ('xyz'):
        t['coordinates'][f'{i}c'] = t['coordinates'][f'{i}']
        del t['coordinates'][f'{i}']
json_dat
with open("primary_corrected_FOV_000.json", 'w') as f:
    json.dump(json_dat, f)
for t in json_dat['tiles']:
    for i in 'yx':
        del t['indices'][f'{i}']
with open("primary_corrected_FOV_000.json", 'w') as f:
    json.dump(json_dat, f)
