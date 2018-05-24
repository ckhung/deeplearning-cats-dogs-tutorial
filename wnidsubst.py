#!/usr/bin/python

import argparse, fileinput, re

parser = argparse.ArgumentParser(
    description='substitute all wnids with their texts',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-w', '--wnid', type=str, help='wnid file')
parser.add_argument('text_files', nargs='*')
args = parser.parse_args()

with open(args.wnid) as f:
    wnid_content = f.readlines()
wnid = {}
for line in wnid_content:
    fields = line.rstrip().split(' ', 1)
    if len(fields)==2 and not re.match(r'^\s*#', line):
        wnid[fields[0]] = fields[1]

for line in fileinput.input(args.text_files):
    line = line.rstrip()
    for k in wnid:
        line = re.sub(r'\b'+k+r'\b', wnid[k], line)
    print line
