#IMPORTS
import numpy as np

class Node:
    """
        Nodes in network. Determines type of node and output value

        Parameters:
            type   (string) : type of node; in:input node ; out:output node; hid:hidden:node
            output (float)  : value to be calculated through feedforward
    """

    def __init__(self,_type):
        self.type = _type
        self.output = 0

class Edge:
    """
        stores information about connection between 2 nodes

        Parameters:
            innov    (int)   : innovation number. to be used in crossover
            in       (int)   : entry node
            out      (int)   : output node
            weight   (float) : value of edge between nodes
            disabled (bool)  : wether edge enabled or not
    """

    def __init__(self,_innov,_in_node,_out_node):
        
        self.in_node = _in_node
        self.out_node = _out_node
        self.weight = np.random.uniform(-1,1)
        self.disabled = False
        self.adj = []

class Genome:
    """
        representation of network as nodes and edges

        Parameters:
            input_nodes  (int)  : number of input nodes
            output_nodes (int)  : number of output nodes
            init_pop     (bool) : True: init network; False: leave blank will inherit from parents
    """
    

    def __init__(self,n_inputs=1,n_outputs=1,init_pop=False):
        
        self.fitness = 0
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs

        self.nodes = []
        self.edges = []

        if init_pop == True:

            #creating input nodes#
            for i in range(0,self.n_inputs,1):
                node = Node(_type="in")
                self.nodes.append(node)
            
            #creating output nodes#
            for i in range(0,self.n_outputs,1):
                node = Node(_type="out")
                self.nodes.append(node)
            
            #initialising connections#
            count = 0

            for i in range(0,self.n_inputs,1):
                for j in range(self.n_inputs,self.n_inputs+self.n_outputs,1):
                    edge = Edge(_innov=count, _in_node=i, _out_node=j)
                    self.edges.append(edge)
                    count+=1
    
    def feedforward(self,x):
        """
            propagates x through network

            Parameters:
                x (array) : input state
        """

        #setting input node values#
        for i in range(0,self.n_inputs,1):
            self.nodes[i].output = x[i]
        
        #updating hidden layer values#
        for i in range(self.n_inputs+self.n_outputs,len(self.nodes),1):
            #looping through connections#
            total = 0
            for j in range(0,len(self.edges),1):

                if (self.edges[j].out_node == i) and (self.edges[j].disabled == False):
                    index = self.edges[j].in_node
                    w = self.edges[j].weight
                    a = self.nodes[index].output
                    temp = w*a
                    total+=temp
            
            self.nodes[i].output = self.activation_function(total,func='sigmoid')
        
        #updating output nodes#
        for i in range(self.n_inputs,self.n_inputs+self.n_outputs,1):
            #looping through connections#
            total = 0
            for j in range(0,len(self.edges),1):

                if (self.edges[j].out_node == i) and (self.edges[j].disabled == False):
                    index = self.edges[j].in_node
                    w = self.edges[j].weight
                    a = self.nodes[index].output
                    temp = w*a
                    total+=temp
            
            self.nodes[i].output = self.activation_function(total,func='sigmoid')
        
        #getting output (might be redundant)#
        y = []
        for i in range(self.n_inputs,self.n_inputs+self.n_outputs,1):
            y.append(self.nodes[i].output)
        
        if len(y) == 1:
            return y[0]
        else:
            return y

    ######################################ACTIVATION_FUNCTIONS#################################################
        def sigmoid(self,x):
            value = 1/(1+np.exp(-x))
            return value
        
        def relu(self,x):
            value = np.max(0,x)
            return value
        
        def activation_function(self,x,func='sigmoid'):
            """
                can choose which activation function to use
                (More to be added)

            """
            if func == 'sigmoid':
                return self.sigmoid(x)
            
            if func== 'relu':
                return self.relu(x)
    #############################################################################################################

if __name__=="__main__":

    model = Genome(n_inputs=2,n_outputs=1,init_pop=True)
    
    #testing#
    x = [1,1]
    output = model.feedforward(x)
    print(output)

