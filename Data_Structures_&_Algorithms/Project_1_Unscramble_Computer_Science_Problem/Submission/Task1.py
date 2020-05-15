"""
Read file into texts and calls.
It's ok if you don't understand how to read files.
"""
import csv
with open('texts.csv', 'r') as f:
    reader = csv.reader(f)
    texts = list(reader)

with open('calls.csv', 'r') as f:
    reader = csv.reader(f)
    calls = list(reader)

# Answer
if __name__ == '__main__':

    # Empty list
    no_of_phones = set()
    
    # Store all no. from texts
    for item in texts:
        no_of_phones.add(item[0])
        no_of_phones.add(item[1])
    # Store all no. from calls
    for item in calls:
        no_of_phones.add(item[0])
        no_of_phones.add(item[1])

    # Result
    print("There are %s different telephone numbers in the records." % (len(no_of_phones)))
"""
TASK 1:
How many different telephone numbers are there in the records? 
Print a message:
"There are <count> different telephone numbers in the records."
"""
