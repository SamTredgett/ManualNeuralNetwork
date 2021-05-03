from sklearn.datasets import make_blobs
import numpy as np
import matplotlib.pyplot as plt

class Operation():
    '''
        Objects of type Operation
    '''
    def __init__(self, input_nodes=[]):
        self.input_nodes=input_nodes
        self.output_nodes=[]

        for node in input_nodes:
            node.output_nodes.append(self)

        _default_graph.operations.append(self)
    def compute(self):
        pass
    
class add(Operation):
    '''
        This implements simple addition for the objects we pass into it
    '''
    def __init__(self, x, y):
        super().__init__([x,y])

    def compute(self, x_var, y_var):
        self.inputs = [x_var, y_var]
        return x_var + y_var

class multiply(Operation):
    '''
        This implemenets simple multiplication for objects we pass into it
    '''
    def __init__(self, x, y):
        super().__init__([x,y])

    def compute(self, x_var,y_var):
        self.inputs = [x_var, y_var]
        return x_var * y_var


class matmul(Operation):
    ''' Function for basic matrix multipliction '''
    def __init__(self, x, y):
        super().__init__([x,y])

    def compute(self, x_var, y_var):
        self.inputs = [x_var, y_var]

        return x_var.dot(y_var)

class Placeholder():
    def __init__(self):
        self.output_nodes=[]
        _default_graph.placeholders.append(self)

class Variable(): 
    ''' allows us to create very loosely defined variables'''
    def __init__(self, initial_value=None):
        self.value=initial_value
        self.output_nodes = []
        _default_graph.variables.append(self)

class Graph():
    ''' A broader object we'll use for the majority of operations'''
    def __init__(self):
        self.operations = []
        self.placeholders = []
        self.variables = []

    def set_as_default(self):
        global _default_graph
        _default_graph = self




def traverse_postorder(operation):
    """
    PostOrder Traversal of Nodes. Basically makes sure computations are done in
    the correct order (Ax first, then Ax + b).
    """

    nodes_postorder = []
    def recurse(node):
        if isinstance(node, Operation):
            for input_node in node.input_nodes:
                recurse(input_node)
        nodes_postorder.append(node)
    recurse(operation)
    return nodes_postorder

class Session():

    ''' Sets up the order in which information is passed between nodes '''
    def run(self,operation, feed_dict={}):
        nodes_postorder = traverse_postorder(operation)
        for node in nodes_postorder:
            if type(node) == Placeholder:
                node.output = feed_dict[node]
            elif type(node) == Variable:
                node.output = node.value
            else: 
                # OPERATION 

                node.inputs = [input_node.output for input_node in node.input_nodes]
                node.output = node.compute(*node.inputs)
            if type(node.output) == list:
                node.output = np.array(node.output)
        return operation.output

''' This was some test code for earlier on
# sess = Session()

# g = Graph()

# g.set_as_default()

# A = Variable([[10,20],[30,40]])
# b = Variable([1,1])

# x = Placeholder()

# y = matmul(A, x)

# z = add(y,b)

# result = sess.run(operation=z, feed_dict={x:10})

# print(result)

'''
# Classification Section
# Activation Function


def sigmoid(z):
    ''' Defining the sigmoid activation function '''
    return 1 / (1 + np.exp(-z))

sample_z = np.linspace(-10,10,100)
sample_a = sigmoid(sample_z)

plt.plot(sample_z,sample_a)
plt.show()

class Sigmoid(Operation):
    ''' Setting up Sigmoid as a class'''
    def __init__(self, z):
        super().__init__([z])
    def compute(self,z_val):
        return 1 / (1 + np.exp(-z_val))

data = make_blobs(n_samples=50, n_features=2,centers=2,random_state=75)

features = data[0]
labels = data[1]

plt.scatter(features[:,0],features[:,1])
plt.show()


x = np.linspace(0,11,10)
y = -x

plt.scatter(features[:,0],features[:,1],c=labels,cmap='coolwarm')
plt.plot(x,y)
plt.show()


# End of the program running bits to get things finally going below:

g = Graph()
g.set_as_default()

x = Placeholder()
w = Variable([1,1])
b = Variable(-5)
z = add(matmul(w,x),b)
a = Sigmoid(z)

sess = Session()

print(sess.run(operation=a, feed_dict={x:[8,10]}))
