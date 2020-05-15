


# Problem 1

def binary_search(start, end, number):
    
    mid = (start + end) / 2
    
    # Base Case 
    
    # Outting 100 digits is enough to find the answer
    if (mid * mid == number) or (len(str(mid)) > 100):

        return int(mid)
    
    elif mid * mid > number:
        return binary_search(start, mid, number)
    
    else:
        return binary_search(mid, end, number)
        


def sqrt(number):
    """
    Calculate the floored square root of a number

    Args:
       number(int): Number to find the floored squared root
    Returns:
       int: Floored Square Root
    """
    
    # handle Edge case
    if (number < 0):
        return None
    
    return binary_search(0, number, number)



if __name__ == '__main__':

    print ("Pass" if  (3 == sqrt(9)) else "Fail")
    print ("Pass" if  (0 == sqrt(0)) else "Fail")
    print ("Pass" if  (4 == sqrt(16)) else "Fail")
    print ("Pass" if  (1 == sqrt(1)) else "Fail")
    print ("Pass" if  (5 == sqrt(27)) else "Fail")
    # Egde Case
    print ("Pass" if  (None == sqrt(-5)) else "Fail")
