# Problem 6


class Node:
    def __init__(self, value):
        self.value = value
        self.next = None

    def __repr__(self):
        return str(self.value)


class LinkedList:
    def __init__(self):
        self.head = None

    def __str__(self):
        cur_head = self.head
        out_string = ""
        while cur_head:
            out_string += str(cur_head.value) + " -> "
            cur_head = cur_head.next
        return out_string

    def get_prev_node(self, ref_node):
        current = self.head
        while (current and current.next != ref_node):
            current = current.next
        return current

    def insert_at_end(self, new_node):
        if self.head is None:
            self.head = new_node
        else:
            current = self.head
            while current.next is not None:
                current = current.next
            current.next = new_node
            
    def duplicate(self):
        copy = LinkedList()
        current = self.head
        while current:
            node = Node(current.value)
            copy.insert_at_end(node)
            current = current.next
        return copy

    def remove(self, node):
        prev_node = self.get_prev_node(node)
        if prev_node is None:
            self.head = self.head.next
        else:
            prev_node.next = node.next
        
    def append(self, value):

        if self.head is None:
            self.head = Node(value)
            return

        node = self.head
        while node.next:
            node = node.next
        node.next = Node(value)

    def size(self):
        size = 0
        node = self.head
        while node:
            size += 1
            node = node.next
        return size



def remove_duplicate(llist):
    current1 = llist.head
    while current1:
        current2 = current1.next
        data = current1.value
        while current2:
            temp = current2
            current2 = current2.next
            if temp.value == data:
                llist.remove(temp)
        current1 = current1.next

'''
def remove_duplicate(llist):
    node = llist.head
    while node is not None:
        lead = node
        while node.next and node.next.value == lead.value:
            node = node.next
        node = lead.next = node.next
    return llist
 '''
 

def union(llist_1, llist_2):
    
    # Your Solution Here
    if llist_1.head is None:
        union = llist_2.duplicate()
        remove_duplicate(union)
        return union
    
    if llist_2 is None:
        union = llist_1.duplicate()
        remove_duplicate(union)
        return union
    
    union = llist_1.duplicate()
    current_1 = union.head
    while current_1.next is not None:
        current_1 = current_1.next
    llist_2_copy = llist_2.duplicate()
    current_1.next = llist_2_copy.head
    remove_duplicate(union)
    
    return union


def intersection(llist_1, llist_2):
    
    # Your Solution Here
    if llist_1.head is None or llist_2.head is None:
        return LinkedList()
    
    intersect = LinkedList()
    current_1 = llist_1.head
    
    while current_1:
        current_2 = llist_2.head
        current_1_value = current_1.value
        current_1 = current_1.next
        while current_2:
            current_2_value = current_2.value
            current_2 = current_2.next
            if (current_1_value == current_2_value):
                # Append 
                intersect.append(current_1_value)
                break
    # Remove duplicate 
    remove_duplicate(intersect)
    return intersect


# Test case 1

linked_list_1 = LinkedList()
linked_list_2 = LinkedList()

element_1 = [3,2,4,35,6,65,6,4,3,21]
element_2 = [6,32,4,9,6,1,11,21,1]

for i in element_1:
    linked_list_1.append(i)

for i in element_2:
    linked_list_2.append(i)

print (union(linked_list_1,linked_list_2))
print (intersection(linked_list_1,linked_list_2))


# Test case 2

linked_list_3 = LinkedList()
linked_list_4 = LinkedList()

element_1 = [3,2,4,35,6,65,6,4,3,23]
element_2 = [1,7,8,9,11,21,1]

for i in element_1:
    linked_list_3.append(i)

for i in element_2:
    linked_list_4.append(i)

print (union(linked_list_3,linked_list_4))
print (intersection(linked_list_3,linked_list_4))


# Test case 3

linked_list_5 = LinkedList()
linked_list_6 = LinkedList()

element_1 = [3,2,4,35,6,65,6,4,3,23]
element_2 = []

for i in element_1:
    linked_list_5.append(i)

for i in element_2:
    linked_list_6.append(i)

print (union(linked_list_5,linked_list_6))
print (intersection(linked_list_5,linked_list_6))




