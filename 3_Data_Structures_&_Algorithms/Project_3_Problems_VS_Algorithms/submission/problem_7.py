

# Problem 7


from collections import defaultdict

# A RouteTrie will store our routes and their associated handlers
class RouteTrie:
    
    
    def __init__(self, root_handler):
        # Initialize the trie with an root node and a handler, this is the root path or home page node
        self.root = RouteTrieNode(root_handler)


    def insert(self, path_components, handler):
        # Similar to our previous example you will want to recursively add nodes
        # Make sure you assign the handler to only the leaf (deepest) node of this path
        node = self.root
        for c in path_components:
            node = node.children[c]
        node.handler = handler


    def find(self, path_components):
        # Starting at the root, navigate the Trie to find a match for this path
        # Return the handler for a match, or None for no match
        node = self.root
        for c in path_components:
            if c not in node.children:
                return None
            node = node.children[c]
        return node.handler


# A RouteTrieNode will be similar to our autocomplete TrieNode... with one additional element, a handler.
class RouteTrieNode:
    
    
    def __init__(self, handler=None):
        # Initialize the node with children as before, plus a handler
        self.children = defaultdict(RouteTrieNode)
        self.handler = handler


    def insert(self):
        # Insert the node as before
        pass


# The Router wraps the Trie and handler
class Router:
    
    
    def __init__(self, root_handler, not_found_handler):
        # Create a new RouteTrie for holding our routes
        # You could also add a handler for 404 page not found responses as well!
        self.trie = RouteTrie(root_handler)
        self.not_found_handler = not_found_handler


    def add_handler(self, path, handler):
        # Add a handler for a path
        # You will need to split the path and pass the pass parts
        # as a list to the RouteTrie
        path_components = Router.split_path(path)
        self.trie.insert(path_components, handler)


    def lookup(self, path):
        # lookup path (by parts) and return the associated handler
        # you can return None if it's not found or
        # return the "not found" handler if you added one
        # bonus points if a path works with and without a trailing slash
        # e.g. /about and /about/ both return the /about handler
        path_components = Router.split_path(path)
        handler = self.trie.find(path_components)
        return handler if handler else self.not_found_handler


    def split_path(path):
        # you need to split the path into parts for 
        # both the add_handler and loopup functions,
        # so it should be placed in a function here
        return [c for c in path.split('/') if c]



if __name__ == '__main__':
    
    
    router = Router("root handler", "not found handler")
    router.add_handler("/home/about", "about handler")

    print(router.lookup("/"))
    # root handler

    print(router.lookup("/home"))
    # not found handler

    print(router.lookup("/home/about"))
    # about handler

    print(router.lookup("/home/about/"))
    # about handler

    print(router.lookup("/home/about/me"))
    # not found handler
    
    
    
    
    
    
    
    