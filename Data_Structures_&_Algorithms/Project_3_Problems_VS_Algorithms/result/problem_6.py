


# Problem 6



def get_min_max(ints):
    """
    Return a tuple(min, max) out of list of unsorted integers.

    Args:
       ints(list): list of integers containing one or more integers
    """
    
    # Empty list 
    if not ints:
        return None
    
    # Edge Case: Single element lists
    if len(ints) == 1:
        return (ints[0], ints[0])
    
    max_num = ints[0]
    min_num = ints[0]
   
    for num in ints:
        
        # Max
        if num > max_num:
            max_num = num
        
        # Min 
        if num < min_num:
            min_num = num 
        
    return (min_num, max_num)

## Example Test Case of Ten Integers
import random

if __name__ == '__main__':

    
    l = [i for i in range(0, 10)]  # a list containing 0 - 9
    random.shuffle(l)
    print ("Pass" if ((0, 9) == get_min_max(l)) else "Fail")
    
    
    l = [i for i in range(0, 20)]  # a list containing 0 - 19
    random.shuffle(l)
    print ("Pass" if ((0, 19) == get_min_max(l)) else "Fail")
    
    l = [i for i in range(0, 100)]  # a list containing 0 - 99
    random.shuffle(l)
    print ("Pass" if ((0, 99) == get_min_max(l)) else "Fail")
    
    l = [i for i in range(0, 1)]  # a list containing 0
    random.shuffle(l)
    print ("Pass" if ((0, 0) == get_min_max(l)) else "Fail")
    
    l = ""
    print ("Pass" if (None == get_min_max(l)) else "Fail")
    
    
    
    
    
    
    
    
    