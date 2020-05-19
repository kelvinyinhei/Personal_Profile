




# Problem 2 


def findPivot(arr, low, high): 
      

    if high < low: 
        return -1
    if high == low: 
        return low 
      
    mid = int((low + high)/2) 
    # base cases 
    if mid < high and arr[mid] > arr[mid + 1]: 
        return mid 
    if mid > low and arr[mid] < arr[mid - 1]: 
        return (mid-1) 
    
    if arr[low] >= arr[mid]: 
        return findPivot(arr, low, mid-1) 
    else:
        return findPivot(arr, mid + 1, high) 



def binary_search(arr, low, high, key):

    if high < low:
        return -1
    
    mid = int((low + high)/2)
    
    # Base Case
    if key == arr[mid]:
        return mid
    
    elif key > arr[mid]:
        return binary_search(arr, (mid+1), high, key)
    
    else:
        return binary_search(arr, low, (mid - 1), key)


    
def rotated_array_search(input_list, number):
    """
    Find the index by searching in a rotated sorted array

    Args:
       input_list(array), number(int): Input array to search and the target
    Returns:
       int: Index or -1
    """
    
    pivot = findPivot(input_list, 0, len(input_list)-1)

    if pivot == -1: 
        return binary_search(input_list, 0, len(input_list)-1, number)
  
    if input_list[pivot] == number: 
        return pivot 
    elif input_list[0] <= number: 
        return binary_search(input_list, 0, pivot-1, number)
    else:
        return binary_search(input_list, pivot+1, len(input_list)-1, number)
  
    
def linear_search(input_list, number):
    for index, element in enumerate(input_list):
        if element == number:
            return index
    return -1



def test_function(test_case):
    input_list = test_case[0]
    number = test_case[1]

    if linear_search(input_list, number) == rotated_array_search(input_list, number):
        print("Pass")
    else:
        print("Fail")




if __name__ == '__main__':
    
    test_function([[6, 7, 8, 9, 10, 1, 2, 3, 4], 6])
    test_function([[6, 7, 8, 9, 10, 1, 2, 3, 4], 1])
    test_function([[6, 7, 8, 1, 2, 3, 4], 8])
    test_function([[6, 7, 8, 1, 2, 3, 4], 1])
    test_function([[6, 7, 8, 1, 2, 3, 4], 10])
    
    # Empty list 
    test_function([[], None])
    
    # Unrotated list 
    test_function([[1, 2, 3, 4, 5, 6, 7], -1])

    
    
    
    
    
    
    
