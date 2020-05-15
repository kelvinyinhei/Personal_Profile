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


# Answer Part (A)
if __name__ == '__main__':

    # Store all Calls
    list_receiver_calls_Bangalore = set()
    list_prefix = ["(", "7", "8", "9"]

    for item in calls:
        if (str(item[0])[0:4] == "(080") or (str(item[1])[0] in list_prefix):
            numbers_called = str(item[1])[1:].split(")")[0]
            list_receiver_calls_Bangalore.add(numbers_called)

    # Sort and convert into list
    list_receiver_calls_Bangalore = sorted(list(list_receiver_calls_Bangalore))
    print("The numbers called by people in Bangalore have codes:")
    print("\n".join(list_receiver_calls_Bangalore))

    

# Answer Part (B)
if __name__ == '__main__':

    # Store all Sender Calls
    list_senders_calls_Bangalore = []
    list_receiver_calls_Bangalore = []
    for item in calls:
        if str(item[0])[0:4] == "(080":
            list_senders_calls_Bangalore.append(item[0])
            list_receiver_calls_Bangalore.append(item[1])
            
    # Total calls
    total_numbers_calls_from_bangalore = len(list_senders_calls_Bangalore)

    # Numbers of receiver calls in Bangalore
    list_receiver_calls_Bangalore = [item for item in list_receiver_calls_Bangalore if str(item)[0:4]=="(080"]

    # Print output
    print("%.2f percent of calls from fixed lines in Bangalore are calls to other fixed lines in Bangalore." % ((len(list_receiver_calls_Bangalore)/ total_numbers_calls_from_bangalore)*100))


"""
TASK 3:
(080) is the area code for fixed line telephones in Bangalore.
Fixed line numbers include parentheses, so Bangalore numbers
have the form (080)xxxxxxx.)

Part A: Find all of the area codes and mobile prefixes called by people
in Bangalore.
 - Fixed lines start with an area code enclosed in brackets. The area
   codes vary in length but always begin with 0.
 - Mobile numbers have no parentheses, but have a space in the middle
   of the number to help readability. The prefix of a mobile number
   is its first four digits, and they always start with 7, 8 or 9.
 - Telemarketers' numbers have no parentheses or space, but they start
   with the area code 140.

Print the answer as part of a message:
"The numbers called by people in Bangalore have codes:"
 <list of codes>
The list of codes should be print out one per line in lexicographic order with no duplicates.

Part B: What percentage of calls from fixed lines in Bangalore are made
to fixed lines also in Bangalore? In other words, of all the calls made
from a number starting with "(080)", what percentage of these calls
were made to a number also starting with "(080)"?

Print the answer as a part of a message::
"<percentage> percent of calls from fixed lines in Bangalore are calls
to other fixed lines in Bangalore."
The percentage should have 2 decimal digits
"""
