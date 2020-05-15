
class Node:
    def __init__(self, value=None, key=None):
        self.prev = None
        self.next = None
        self.key = key
        self.value = value


class DoublyLinkedList:
    def __init__(self):
        self.head = None
        self.tail = None
        self.size = 0

    def push_head(self, node):
        """
        Push node to the head of the list.
        :param node: Node to push
        :return: None
        """
        if self.head:
            self.head.prev = node
        else:
            self.tail = node

        node.next = self.head
        node.prev = None
        self.head = node
        self.size += 1

    def pop_tail(self):
        """
        Remove node from end of list
        :return: last Node in list, with next and prev nullified.  Return None if list is empty.
        """
        if not self.tail:
            return None
        saved_tail = self.tail
        if self.tail.prev:  # tail != head
            self.tail.prev.next = None
            self.tail = self.tail.prev
        else:  # tail == head
            self.head = None
            self.tail = None
        saved_tail.prev = None
        self.size -= 1
        return saved_tail

    def remove(self, node):
        """
        Remove node from list.  Assumes node is in list.
        :param node: Node to remove
        :return: Node that was removed, with next and prev nullified
        """
        if node.prev:  # node is not head
            node.prev.next = node.next
        else:  # node is head
            self.head = node.next
        if node.next:  # node is not tail
            node.next.prev = node.prev
        else:
            self.tail = node.prev
        node.next = None
        node.prev = None
        self.size -= 1
        return node


class LRU_Cache:
    def __init__(self, capacity=5):
        self.capacity = capacity
        self.used = DoublyLinkedList()  # head is most-recently used item, tail is least-recently used
        self.map = dict()  # key -> Node, node contains value

    def get(self, key):
        """
        Return value associated with key.
        Return -1 if the key is not in the cache.
        Updates the used list to indicate that the key is most recently used.
        :param key: the key
        :return: value associated with key
        """
        if key not in self.map:
            return -1
        node = self.map[key]
        self.used.remove(node)
        self.used.push_head(node)
        return node.value

    def set(self, key, value):
        """
        Put value into cache.
        If the cache is full, remove the least-recently-used item from the cache before putting the new value in the
        cache.
        :param key: the key
        :param value: the value associated with the key
        :return: None
        """
        if self.capacity == 0:
            return
        if self.capacity == self.used.size:
            node = self.used.pop_tail()
            if node:
                del self.map[node.key]
        if key in self.map:
            node = self.map[key]
            node.value = value
        else:
            node = Node(value, key)
        self.used.push_head(node)
        self.map[key] = node



# Test Case 1
our_cache = LRU_Cache(2)

our_cache.set(1, 10);
our_cache.set(2, 20);

print(our_cache.get(1))       # returns 10
print(our_cache.get(2))       # returns 20


# Test Case 2

our_cache = LRU_Cache(2)

our_cache.set(1, 1);
our_cache.set(2, 2);
our_cache.set(3, 3);
our_cache.set(4, 4);

print(our_cache.get(1))       # returns -1
print(our_cache.get(4))       # returns 4


# Test Case 3

our_cache = LRU_Cache(2)

our_cache.set(1, 1);
our_cache.set(2, 2);

print(our_cache.get(3))      # returns -1 because the cache reached it's capacity and 3 was the least recently used entry





