"""
Read file into texts and calls.
It's ok if you don't understand how to read files.
"""
import csv
import pandas as pd
import numpy as np

with open('texts.csv', 'r') as f:
    reader = csv.reader(f)
    texts = list(reader)

with open('calls.csv', 'r') as f:
    reader = csv.reader(f)
    calls = list(reader)


# Answer
if __name__ == '__main__':

    # Target list 
    list_of_telemarketers = set([item[0] for item in calls])
    list_of_incoming_calls = set([item[1] for item in calls])
    
    list_of_incoming_texts = set([item[0] for item in texts])
    list_of_outgoing_texts = set([item[1] for item in texts])
    
    # Check for existence
    list_of_telemarketers = [i for i in list_of_telemarketers if i not in list_of_incoming_calls]
    list_of_telemarketers = [i for i in list_of_telemarketers if i not in list_of_incoming_texts]
    list_of_telemarketers = [i for i in list_of_telemarketers if i not in list_of_outgoing_texts]

    # Sorting in lexicographic order
    list_of_telemarketers = sorted(list_of_telemarketers)
    
    # output 
    print("These numbers could be telemarketers: ")
    print('\n'.join(list_of_telemarketers))
"""
TASK 4:
The telephone company want to identify numbers that might be doing
telephone marketing. Create a set of possible telemarketers:
these are numbers that make outgoing calls but never send texts,
receive texts or receive incoming calls.

Print a message:
"These numbers could be telemarketers: "
<list of numbers>
The list of numbers should be print out one per line in lexicographic order with no duplicates.
"""

