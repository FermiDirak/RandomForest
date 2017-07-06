class Node:
    def __init__(self, data, depth):
        self.data = data
        self.left = None
        self.right = None
        self.split = None
        self.depth = depth

    def add_left_child(self, node):
        if node.left == None:
            self.left = node

    def add_right_child(self, node):
        if node.right == None:
            self.right = node

    def createNode(self, data, depth):
        self.data = data
        self.split = getRandomSplit(data)



class Tree:

    def __init__(self, dataset, min_depth):
        self.tree = gen_tree(min_depth)

    def get_data(self, node):
        return node.data

    def gen_tree(self, depth):
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

    #gets split where top right are 'right' and bottom left are 'left'. returns left subset of data from split
    def get_left_split(dataset, split):
        return get_split(dataset, split, true)

    #gets split where top right are 'right' and bottom left are 'left'. returns right subset of data from split
    def getRightSplit(dataset, split):
        return get_split(dataset, split, false)

    #get split. pass in true for left and false for right for direction
    def get_split(datset, split, direction):
        split = np.empty([:, dataset.size.m])
        is_x_split = (split[1,0] == 0)
        feature = 0
        if !is_x_split:
            feature = 1
        split_value = split[feature, 0]

        j = 0
        for i in range(0, dataset.size.m):
            current_instance = dataset[feature_index, i]

            if ((direction && current_instance <= split_value) || (!direction && current_instance > split_value)):
                split[:,j] = dataset[:, i]
                j++

        split = split[:,0:j+1]
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
