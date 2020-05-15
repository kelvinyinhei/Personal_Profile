# Problem 5

import hashlib
from datetime import datetime


class Node:
    
    def __init__(self, value):
        self.value = value 
        self.next = None

class Block:

    def __init__(self, timestamp, data, previous_hash):
      self.timestamp = timestamp
      self.data = data
      self.previous_hash = previous_hash
      self.hash = self.calc_hash()
      
    def calc_hash(self):
        sha = hashlib.sha256()
        hash_str = self.data.encode('utf-8')
        sha.update(hash_str)
        return sha.hexdigest()   
     

      
class BlockChain:
    def __init__(self, timestamp, data):
        self.head = Node(Block(timestamp, data, None))
        self.tail = self.head
    
    def add_block(self, timestamp, data):
        new_block = Block(timestamp, data, self.tail.value.hash)
        new_node = Node(new_block)
        self.tail.next = new_node
        self.tail = new_node
        
    def print_blockchain(self):
            node = self.head
            while node:
                print('Timestamp: \t', node.value.timestamp)
                print('Data: \t\t', node.value.data)
                print('SHA256 Hash: \t', node.value.hash)
                print('Prev Hash: \t', node.value.previous_hash, '\n')
                node = node.next

# Test Case 1
print("---------- Test Case 1 -----------")
blockchain = BlockChain(datetime.now(), 'sample testing')
blockchain.print_blockchain()

# Test Case 2
print("---------- Test Case 2 -----------")
blockchain.add_block("12:13 11/7/2019", "Add info")
blockchain.add_block("12:15 11/7/2019", "Add new info")
blockchain.print_blockchain()

# Test Case 3
print("---------- Test Case 3 -----------")
blockchain.add_block("12:30 11/7/2019", "Add final info")
blockchain.print_blockchain()
