#!/usr/bin/env python
"""reducer.py"""

from operator import itemgetter
import sys

prev_year = ""
prev_values = []

# input comes from STDIN
for line in sys.stdin:

    year, words = line.split('\t')

    values = words.split(',')

    # convert value (currently a string) to float
    if (year == prev_year):
	length = len(prev_values)
	for i in range(length):
    	    try:
                prev_values[i] += float(values[i])
    	    except ValueError:
        	prev_values[i] = 0
    else:
	if prev_year != "":
            print prev_year
	    for value in prev_values:
		print ',%s' (value/50.0) # printing the average of 50 states in a year
	prev_year = year
	for value in values:
	    try:
	    	prev_values.append(float(value))
	    except ValueError:
                prev_values.append(0)

