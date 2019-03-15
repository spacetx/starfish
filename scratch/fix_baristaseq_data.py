# what follows is the history that was needed to fix the primaryFOV_000.json file

import json
import re
import sys


def main(input_json: str, output_json: str) -> None:

    def clean_json(string):
        string = re.sub(",[ \t\r\n]+}", "}", string)
        string = re.sub(",[ \t\r\n]+\]", "]", string)

        return string

    # read in the json file
    with open(input_json) as f:
        dat = f.read()
        cleaned = clean_json(dat)
        json_dat = json.loads(cleaned)

    # fix some broken stuff
    for t in json_dat['tiles']:
        for i in ('xyz'):
            t['coordinates'][f'{i}c'] = t['coordinates'][f'{i}']
            del t['coordinates'][f'{i}']

        for i in 'yx':
            del t['indices'][f'{i}']

    with open(output_json, 'w') as f:
        json.dump(json_dat, f)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
