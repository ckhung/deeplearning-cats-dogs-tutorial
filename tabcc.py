#!/usr/bin/python
from __future__ import print_function
import argparse, fileinput, re

parser = argparse.ArgumentParser(
    description='given a csv file, tabulate class counts according to the combination of two fields',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-r', '--row', type=int,
    default=0, help='field # for row indexing')
parser.add_argument('-c', '--col', type=int,
    default=1, help='field # for column indexing')
parser.add_argument('text_files', nargs='*')
args = parser.parse_args()

table = {}
colkeys = []

for line in fileinput.input(args.text_files):
    if re.match(r'^\s*#', line):
        continue
    fields = line.rstrip().split(',')
    n = len(fields)
    if args.row >= n or args.col >= n:
        print('warning: ignoring line: ' + line)
        continue
    rk = fields[args.row]
    ck = fields[args.col]
    if not rk in table:
        table[rk] = {}
    if not ck in colkeys:
        colkeys.append(ck)
    if ck in table[rk]:
        table[rk][ck] += 1
    else:
        table[rk][ck] = 1

colkeys = sorted(colkeys)
for ck in colkeys:
    print(','+ck, end='')
print('')
for rk in sorted(table.keys()):
    print(rk, end='')
    for ck in colkeys:
        if ck not in table[rk]:
            table[rk][ck] = 0
        print(','+str(table[rk][ck]), end='')
    print('')

