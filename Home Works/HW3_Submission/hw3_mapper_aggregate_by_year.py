#!/usr/bin/env python
"""mapper.py"""

import sys

# input comes from STDIN (standard input)
for line in sys.stdin:

    line = line.strip()
    # split the line into words
    words = line.split(',')
    if words[0] == 'PRIMARY_KEY':
	continue
    all_values = ''
    for i in range(3, len(words)):
	all_values += words[i]
	all_values += ','
    print '%s\t%s' % (words[2], all_values) # words[2] contains the year
