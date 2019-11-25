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
    if len(words) < 100:
	continue
    
    print '%s' % (words[19]) # words[19] contains the enrollment in grade 9 to 12
