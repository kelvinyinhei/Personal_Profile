import os 

# Problem 2 
result = []
def find_files(suffix, path):
    """
    Find all files beneath path with file name suffix.

    Note that a path may contain further subdirectories
    and those subdirectories may also contain further subdirectories.

    There are no limit to the depth of the subdirectories can be.

    Args:
      suffix(str): suffix if the file name to be found
      path(str): path of the file system

    Returns:
       a list of paths
    """
    # Base Case
    # if it the file with extension ".c", then append
    if os.path.isfile(path):
        if path.endswith(suffix):
            result.append(path)
    # Else drill down to next level
    else:
        files = os.listdir(path)
        for item in files:
            file_path = os.path.join(path, item)
            find_files(suffix, file_path)
    return result


# Test Case 1 
suffix = ".c"
path = "/Users/yinhei/Library/Mobile Documents/com~apple~CloudDocs/udacity/Data_Structures_and_Algorithms/Ch_2_Data_Structures/Project_Show_Me_The_Data_Structures/result/testdir/"
print(find_files(suffix, path))
print('-----------------------')

# Test Case 2
suffix = ".t"
path = "/Users/yinhei/Library/Mobile Documents/com~apple~CloudDocs/udacity/Data_Structures_and_Algorithms/Ch_2_Data_Structures/Project_Show_Me_The_Data_Structures/result/testdir/"
print(find_files(suffix, path))
print('-----------------------')

# Test Case 3 
suffix = ".zzz"
path = "/Users/yinhei/Library/Mobile Documents/com~apple~CloudDocs/udacity/Data_Structures_and_Algorithms/Ch_2_Data_Structures/Project_Show_Me_The_Data_Structures/result/testdir/"
print(find_files(suffix, path))
print('-----------------------')
