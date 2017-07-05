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
    root = Tree()
