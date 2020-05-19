

# Problem 3 

def quicksort(numlist):
    def helper(left, right):
        if right <= left:
            return
        i = left
        p = right
        while i < p:
            if numlist[i] < numlist[p]:
                i += 1
            else:
                x = numlist[i]
                y = numlist[p - 1]
                pivot = numlist[p]
                numlist[p] = x
                numlist[i] = y
                numlist[p - 1] = pivot
                p -= 1
        helper(left, p - 1)
        helper(p + 1, right)

    helper(0, len(numlist) - 1)


def make_num(digits):

    num = 0
    for d in digits:
        num *= 10
        num += d
    return num


def rearrange_digits(input_list):
    """
    Rearrange Array Elements so as to form two number such that their sum is maximum.

    Args:
       input_list(list): Input List
    Returns:
       (int),(int): Two maximum sums
    """
    quicksort(input_list)
    num1_digits = [input_list[i] for i in range(len(input_list) - 1, -1, -2)]
    num2_digits = [input_list[i] for i in range(len(input_list) - 2, -1, -2)]
    return [make_num(num1_digits), make_num(num2_digits)]



def test_function(test_case):
    output = rearrange_digits(test_case[0])
    solution = test_case[1]
    if sum(output) == sum(solution):
        print("Pass")
    else:
        print("Fail")
        
        
if __name__ == '__main__':
    
    test_function([[7, 4, 5, 0, 6, 3, 8, 10, 9, 2, 1], [1086420, 97531]])
    # Pass

    test_function([[1, 1, 1, 1], [11, 11]])
    # Pass

    test_function([[], [0, 0]])
    # Pass
    
    test_function([[1, 2, 3, 4, 5], [542, 31]])
    # Pass

    test_function([[4, 6, 2, 5, 9, 8], [964, 852]])
    # Pass
    
    
    
    
    
    