class Tree: # passing (object ) into class is no longer needed in python3

    def __init__(self, dataset): # oh god u forgot a colon here
        # not sure what this is interfacing with
        # Node() object?
        self.root = None
        self.left = None
        self.right = None
        self.data = dataset

    def add_head(data):
        if self.root == None:
            self.root = data
    def add_child(data):
        """possible implementation?"""
        pass
    def get_data():
        pass


if __name__ == '__main__':
    data = 5
    root = Tree()
