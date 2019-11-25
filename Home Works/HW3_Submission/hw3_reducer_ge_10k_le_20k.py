#!/usr/bin/env python
"""reducer.py"""

from operator import itemgetter
import sys

count = 0

# input comes from STDIN
for line in sys.stdin:
    # remove leading and trailing whitespace
    value = line.strip()

    # print value
    # convert value (currently a string) to int
    try:
        value = float(value)
    except ValueError:
        continue
    
    if value >= 10000 and value <= 20000:
	# print value
	count = count + 1

print count
