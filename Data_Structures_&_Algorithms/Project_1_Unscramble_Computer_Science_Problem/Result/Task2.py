"""
Read file into texts and calls.
It's ok if you don't understand how to read files
"""
import pandas as pd
import numpy as np
import csv
with open('texts.csv', 'r') as f:
    reader = csv.reader(f)
    texts = list(reader)

with open('calls.csv', 'r') as f:
    reader = csv.reader(f)
    calls = list(reader)


# Answer

if __name__ == '__main__':


    # Store in a dict
    dict_calls = {}
    for item in calls:
        # In Calls
        if item[0] not in dict_calls:
            dict_calls[item[0]] = item[3]
        else:
            val = int(dict_calls[item[0]]) + int(item[3])
            dict_calls[item[0]] = val
        # Out Calls
        if item[1] not in dict_calls:
            dict_calls[item[1]] = item[3]
        else:
            val = int(dict_calls[item[1]]) + int(item[3])
            dict_calls[item[1]] = val

    # Convert dict into list with tuple
    list_of_calls = []
    for key in dict_calls:
        temp = (key , dict_calls[key])
        list_of_calls.append(temp)
    
    # Sorting
    max_one = max(list_of_calls, key=lambda item: int(item[1]))
    print("%s spent the longest time, %s seconds, on the phone during September 2016." % (max_one[0], max_one[1]))

    # Method 2
    #list_of_calls = sorted(list_of_calls, key=lambda tup: float(tup[1]), reverse=True)
    #print("%s spent the longest time, %s seconds, on the phone during September 2016." % (list_of_calls[0][0], list_of_calls[0][1]))


"""
TASK 2: Which telephone number spent the longest time on the phone
during the period? Don't forget that time spent answering a call is
also time spent on the phone.
Print a message:
"<telephone number> spent the longest time, <total time> seconds, on the phone during 
September 2016.".
"""

