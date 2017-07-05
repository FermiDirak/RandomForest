class Node:
    def __init__(self, data=None):
        self.data = data
        self.left = None
        self.right = None


class Tree: # passing (object ) into class is no longer needed in python3

    def __init__(self, dataset, minDepth, depth = 3):
        self.root = None
        self.left = None
        self.right = None
        self.data = dataset
        self.depth = depth

    def add_head(self, data):
        if self.root == None:
            self.root = data #unsafe lol
    def add_left_child(self, data):
        """possible implementation?"""
        pass
    def add_right_child(self, data):
        """possible implementation?"""
        pass
    def get_data(self, node):
        return node.data
    def gen_tree(self, depth=self.depth):
        """ builds full tree recursively """
        if depth == 1:
            return Node()
        self.root = Node()
        self.root.left = gen_tree(depth-1)
        self.root.right = gen_tree(depth-1)
        return self.root




    #gets a random split point for the dataset
    def getRandomSplit(dataset):
        split = np.transpose(np.matrix(np.zeros(2)))
        coordN = np.round(np.random.rand())
        coordM = np.floor(dataset.size.m * np.random.rand())

        split[coordN, 0] = dataset[coordN + 1, coordM]

        return split

   #get the best split point for dataset
    def getBestGiniSplit(dataset, labelsCount):
        return 0

    #calculates gini value of a given dataset
    def calcGini(histogram, labelsCount):
        gini = 0
        for i in range(0, histogram.size.m)
            gini += histogram[0, i] * histogram[0, i]
        gini = 1 - gini
        return gini


    #returns a 1 * labelCount matrix of histogram data
    def getHistogram(dataset, labelsCount):
        histogram = np.matrix(np.zeros(labelsCount))
        for i in range(dataset.size.m)
            j = dataset[0, i]
            histogram[0, j] += 1
        return histogram


if __name__ == '__main__':
    # toy tree demo
    root = Tree()
