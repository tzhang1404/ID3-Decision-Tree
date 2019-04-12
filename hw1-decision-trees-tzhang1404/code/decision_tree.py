import numpy as np
from math import log2
import copy

class Tree():
    def __init__(self, value=None, attribute_name="root", attribute_index=None, branches=None):
        """
        This class implements a tree structure with multiple branches at each node.
        If self.branches is an empty list, this is a leaf node and what is contained in
        self.value is the predicted class.

        The defaults for this are for a root node in the tree.

        Arguments:
            branches (list): List of Tree classes. Used to traverse the tree. In a
                binary decision tree, the length of this list is either 2 (for left and
                right branches) or 0 (at a leaf node).
            attribute_name (str): Contains name of attribute that the tree splits the data
                on. Used for visualization (see `DecisionTree.visualize`).
            attribute_index (float): Contains the  index of the feature vector for the
                given attribute. Should match with self.attribute_name.
            value (number): Contains the value that data should be compared to along the
                given attribute.
        """
        self.branches = [] if branches is None else branches
        self.attribute_name = attribute_name
        self.attribute_index = attribute_index
        self.value = value

class DecisionTree():
    def __init__(self, attribute_names):
        """
        TODO: Implement this class.

        This class implements a binary decision tree learner for examples with
        categorical attributes. Use the ID3 algorithm for implementing the Decision
        Tree: https://en.wikipedia.org/wiki/ID3_algorithm

        A decision tree is a machine learning model that fits data with a tree
        structure. Each branching point along the tree marks a decision (e.g.
        today is sunny or today is not sunny). Data is filtered by the value of
        each attribute to the next level of the tree. At the next level, the process
        starts again with the remaining attributes, recursing on the filtered data.

        Which attributes to split on at each point in the tree are decided by the
        information gain of a specific attribute.

        Here, you will implement a binary decision tree that uses the ID3 algorithm.
        Your decision tree will be contained in `self.tree`, which consists of
        nested Tree classes (see above).

        Args:
            attribute_names (list): list of strings containing the attribute names for
                each feature (e.g. chocolatey, good_grades, etc.)

        """
        self.attribute_names = attribute_names
        self.tree = None
        return

    def _check_input(self, features):
        if features.shape[1] != len(self.attribute_names):
            raise ValueError(
                "Number of features and number of attribute names must match!"
            )

    def choose_attribute(self, features, targets, attribute_names):
        print("inside choose")
        currBest = attribute_names[0]
        currMaxInfoGain = 0
        index = 0;
        for attribute_index in range(len(attribute_names)):
            newInfoGain = information_gain(features, attribute_index, targets)
            print(newInfoGain)
            if newInfoGain > currMaxInfoGain:
                currMaxInfoGain = newInfoGain
                currBest = attribute_names[attribute_index]
                index = attribute_index
        return currBest, index, currMaxInfoGain

    #load the rows in features that have a specific value for a specific attribute
    def split_rows(self, features, targets, value, attribute_index, attribute_names):
        newFeatures = []
        newTargets = [] 

        for row in range(len(targets)):
            if features[row][attribute_index] == value:
                newTargets.append(targets[row])
                newFeatures.append(features[row])


        #cast back to numpy array
        newFeatures = np.asarray(newFeatures)
        newTargets = np.asarray(newTargets)

        return newFeatures, newTargets

    def getDominant(self, targets):
        trueCount = 0
        falseCount = 0

        for value in targets:
            if value == 1:
                trueCount += 1
            else:
                falseCount += 1

        if trueCount >= falseCount:
            return 1
        else:
            return 0


    def fit(self, features, targets):
        """
        Takes in the features as a numpy array and fits a decision tree to the targets.

        Args:
            features (np.array): numpy array of size NxF containing features, where N is
                number of examples and F is number of features.
            targets (np.array): numpy array containing class labels for each of the N
                examples.
        """

        #return the tree by setting it to the class field 
        self._check_input(features)
        print("enter fit", self.attribute_names)
        self.tree = self.ID3(features, targets, self.attribute_names)
                
    def ID3(self, features, targets, attribute_names):
        if np.sum(targets) == targets.shape[0]:
            #all positive
            return Tree(value = 1)
        if np.sum(targets) == 0:
            #all negative
            return Tree(value = 0)

        if len(attribute_names) == 0:
            return Tree(value = self.getDominant(targets))
        #get the attribute with the best info gain
        chosenAttribute, index, maxGain = self.choose_attribute(features, targets, attribute_names)

        #create the treenode with the attribute
        newTree = Tree(attribute_name = chosenAttribute, attribute_index = index)

        # values should be 1 or 0 (true->left branch, false->right branch)
        for val in [1, 0]:
            newFeatures, newTargets = self.split_rows(features, targets, val, index, attribute_names)
            if len(newFeatures) == 0:
                #this is a tree node so get the dominant value and store it as its value
                dominateValue = self.getDominant(targets)
                return Tree(value = dominateValue)
            else:
                #this is not a leaf node so create the tree node
                #remove the used attribute from the list nad from the feature list
                newAttributes = copy.deepcopy(attribute_names)
                newAttributes.remove(chosenAttribute)
                nF = np.concatenate((newFeatures[:, :index], newFeatures[:, index + 1:]), axis = 1)
                #add the new node to the branch
                newTree.branches.append(self.ID3(nF, newTargets, newAttributes));

        return newTree

    def predict(self, features):
        """
        Takes in features as a numpy array and predicts classes for each point using
        the trained model.

        Args:
            features (np.array): numpy array of size NxF containing features, where N is
                number of examples and F is number of features.
        """
        self._check_input(features)
        #self.visualize()
        #initialize result array
        result = np.zeros(shape = len(features))

        for row in range(features.shape[0]):
            currNode = self.tree
            #self.visualize();
            while currNode.attribute_name != "root":
                #the node attribute is true for this example
                index = -1;
                for i in range(len(self.attribute_names)):
                    if self.attribute_names[i] == currNode.attribute_name:
                        index = i
                        break
                if features[row][index] == 1:
                    currNode = currNode.branches[0]
                #the node attribute is false
                else:
                    currNode = currNode.branches[1]

            #outside the while loop, meaning that we are at a leaf node
            result[row] = currNode.value
        return result


    def _visualize_helper(self, tree, level):
        """
        Helper function for visualize a decision tree at a given level of recursion.
        """
        tab_level = "  " * level
        val = tree.value if tree.value is not None else 0
        print("%d: %s%s == %f" % (level, tab_level, tree.attribute_name, val))

    def visualize(self, branch=None, level=0):
        """
        Visualization of a decision tree. Implemented for you to check your work and to
        use as an example of how to use the given classes to implement your decision
        tree.
        """
        if not branch:
            branch = self.tree
        self._visualize_helper(branch, level)

        for branch in branch.branches:
            self.visualize(branch, level+1)

def get_entropy(targets):
    entropy = 0.0
    # the total number of elements
    totalElement = len(targets);
    if totalElement == 0:
        return 0; 
    positiveClass = 0;
    negativeClass = 0;

    for row in range(totalElement):
        if targets[row] == 1:
            positiveClass += 1
        else:
            negativeClass += 1

    posProb = positiveClass/totalElement
    negProb = negativeClass/totalElement


    if posProb == 0 or negProb == 0:
        return 0


    entropy = -(posProb) * log2(posProb) + -(negProb) * log2(negProb)
    return entropy

        
def information_gain(features, attribute_index, targets):
    """
    TODO: Implement me!

    Information gain is how a decision tree makes decisions on how to create
    split points in the tree. Information gain is measured in terms of entropy.
    The goal of a decision tree is to decrease entropy at each split point as much as
    possible. This function should work perfectly or your decision tree will not work
    properly.

    Information gain is a central concept in many machine learning algorithms. In
    decision trees, it captures how effective splitting the tree on a specific attribute
    will be for the goal of classifying the training data correctly. Consider
    data points S and an attribute A. S is split into two data points given binary A:

        S(A == 0) and S(A == 1)

    Together, the two subsets make up S. If A was an attribute perfectly correlated with
    the class of each data point in S, then all points in a given subset will have the
    same class. Clearly, in this case, we want something that captures that A is a good
    attribute to use in the decision tree. This something is information gain. Formally:

        IG(S,A) = H(S) - H(S|A)

    where H is information entropy. Recall that entropy captures how orderly or chaotic
    a system is. A system that is very chaotic will evenly distribute probabilities to
    all outcomes (e.g. 50% chance of class 0, 50% chance of class 1). Machine learning
    algorithms work to decrease entropy, as that is the only way to make predictions
    that are accurate on testing data. Formally, H is defined as:

        H(S) = sum_{c in (classes in S)} -p(c) * log_2 p(c)

    To elaborate: for each class in S, you compute its prior probability p(c):

        (# of elements of class c in S) / (total # of elements in S)

    Then you compute the term for this class:

        -p(c) * log_2 p(c)

    Then compute the sum across all classes. The final number is the entropy. To gain
    more intution about entropy, consider the following - what does H(S) = 0 tell you
    about S?

    Information gain is an extension of entropy. The equation for information gain
    involves comparing the entropy of the set and the entropy of the set when conditioned
    on selecting for a single attribute (e.g. S(A == 0)).

    For more details: https://en.wikipedia.org/wiki/ID3_algorithm#The_ID3_metrics

    Args:
        features (np.array): numpy array containing features for each example.
        attribute_index (int): which column of features to take when computing the
            information gain
        targets (np.array): numpy array containing labels corresponding to each example.

    Output:
        information_gain (float): information gain if the features were split on the
            attribute_index.
    """
    normalEntropy = get_entropy(targets)
    GivenAttriTrue = 0.0;
    GivenAttriFalse = 0.0;

    trueAttributeArray = []
    falseAttributeArray = []

    
    for row in range(features.shape[0]):
        if features[row][attribute_index] == 1:
            GivenAttriTrue += 1
            trueAttributeArray.append(targets[row])
        else:
            GivenAttriFalse += 1
            falseAttributeArray.append(targets[row])


    if GivenAttriTrue == 0:
        probGivenTrue = 0
        probGivenFalse = 1
    elif GivenAttriFalse == 0:
        probGivenFalse = 0
        probGivenTrue = 1
    else:
        probGivenTrue = GivenAttriTrue / features.shape[0]
        probGivenFalse = GivenAttriFalse / features.shape[0]

    
    conditionalEntropy = (probGivenTrue) * get_entropy(trueAttributeArray) + (probGivenFalse) * get_entropy(falseAttributeArray)
    
    information_gain = normalEntropy - conditionalEntropy


    return information_gain



    

if __name__ == '__main__':
    # construct a fake tree
    attribute_names = ['larry', 'curly', 'moe']
    decision_tree = DecisionTree(attribute_names=attribute_names)
    while len(attribute_names) > 0:
        attribute_name = attribute_names[0]
        if not decision_tree.tree:
            decision_tree.tree = Tree(
                attribute_name=attribute_name,
                attribute_index=decision_tree.attribute_names.index(attribute_name),
                value=0,
                branches=[]
            )
        else:
            decision_tree.tree.branches.append(
                Tree(
                    attribute_name=attribute_name,
                    attribute_index=decision_tree.attribute_names.index(attribute_name),
                    value=0,
                    branches=[]
                )
            )
        attribute_names.remove(attribute_name)
    decision_tree.visualize()
