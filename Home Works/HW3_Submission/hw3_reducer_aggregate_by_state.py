#!/usr/bin/env python
"""reducer.py"""

from operator import itemgetter
import sys

prev_state = ""
prev_values = []

# input comes from STDIN
for line in sys.stdin:

    state, words = line.split('\t')

    values = words.split(',')

    # convert value (currently a string) to float
    if (state == prev_state):
	length = len(prev_values)
	for i in range(length):
    	    try:
                prev_values[i] += float(values[i])
    	    except ValueError:
        	prev_values[i] = 0
    else:
	if prev_state != "":
            print prev_state
	    for value in prev_values:
		print ',%s' (value)
	prev_state = state
	for value in values:
	    try:
	    	prev_values.append(float(value))
	    except ValueError:
                prev_values.append(0)

