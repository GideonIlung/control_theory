#IMPORTS
import numpy as np
import zipfile
import os
import pandas as pd

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
            enabled (bool)  : wether edge enabled or not
    """

    def __init__(self,_innov,_in_node,_out_node,_weight = None,_enabled = 1):
        
        self.in_node = _in_node
        self.out_node = _out_node

        if _weight == None:
            self.weight = np.random.uniform(-1,1)
        else:
            self.weight = _weight
        
        self.enabled = _enabled
        self.innov = _innov

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
        self.n_hidden = 0
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
    
            count = 0

            for i in range(0,self.n_inputs,1):
                for j in range(self.n_inputs,self.n_inputs+self.n_outputs,1):
                    edge = Edge(_innov=count, _in_node=i, _out_node=j)
                    self.edges.append(edge)
                    count+=1
    
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

                if (self.edges[j].out_node == i) and (self.edges[j].enabled == 1):
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

                if (self.edges[j].out_node == i) and (self.edges[j].enabled == 1):
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
    #################################################processing functions ####################################
    def print_genome(self):
        """
            prints Genome information
        """
    
        print("Node information \n===============================================")

        for i in range(0,len(self.nodes),1):
            print('Node {} : {} '.format(i,self.nodes[i].type))
        
        print("===============================================")

        print("Edges information \n===============================================")

        for i in range(0,len(self.edges),1):
            print("Edge {}; innov {}; in {}; out {}; weight {}; enabled {}".format(i,self.edges[i].innov,self.edges[i].in_node,self.edges[i].out_node,self.edges[i].weight,self.edges[i].enabled))
        print("===============================================")
    
    def flatten(self):
        """
            flattens model and returns info as lists
        """

        #getting node type#
        node_type = []
        for i in range(0,len(self.nodes),1):
            node_type.append(self.nodes[i].type)

        #getting edge info#
        innov_list = []
        in_list = []
        out_list = []
        w_list = []
        enabled_list = []

        for x in self.edges:
            innov_list.append(x.innov)
            in_list.append(x.in_node)
            out_list.append(x.out_node)
            w_list.append(x.weight)
            enabled_list.append(x.enabled)
        
        return node_type,innov_list,in_list,out_list,w_list,enabled_list

    def save_model(self,filename='output'):
        """
            Saves model to zip file

            Parameters:
                filename (string) : name of zipfile
        """
        name = filename + '.zip'
        node_type,innov_list,in_list,out_list,w_list,enabled_list = self.flatten()

        #saving nodes#
        nodefile = open('node.txt','w')
        lines = []
        for x in node_type:
            string = x + '\n'
            lines.append(string)
        nodefile.writelines(lines)
        nodefile.close()

        #saving innovation numbers#
        d = {'values':innov_list}
        data = pd.DataFrame(d)
        data.to_csv('innov.csv')

        #saving in nodes number#
        d = {'values':in_list}
        data = pd.DataFrame(d)
        data.to_csv('in.csv')

        #saving out nodes#
        d = {'values':out_list}
        data = pd.DataFrame(d)
        data.to_csv('out.csv')

        #saving weights
        d = {'values':w_list}
        data = pd.DataFrame(d)
        data.to_csv('weight.csv')

        #saving enabled info#
        d = {'values':enabled_list}
        data = pd.DataFrame(d)
        data.to_csv('enabled.csv')

        outzip = zipfile.ZipFile(name,'w')
        outzip.write('node.txt')
        os.remove('node.txt')

        outzip.write('innov.csv')
        os.remove('innov.csv')

        outzip.write('in.csv')
        os.remove('in.csv')

        outzip.write('out.csv')
        os.remove('out.csv')

        outzip.write('weight.csv')
        os.remove('weight.csv')

        outzip.write('enabled.csv')
        os.remove('enabled.csv')
        outzip.close()

    def load_model(self,filename='output.zip'):
        """
            Loads saved model
        """

        with zipfile.ZipFile(filename,'r') as parent_file:
            
            
            self.nodes = []
            self.edges = []
            self.n_inputs = 0
            self.n_hidden = 0
            self.n_outputs = 0

            #getting node info#
            nodefile = parent_file.open('node.txt')
            Lines = nodefile.readlines()

            for line in Lines:
                sent = line.strip()
                data = sent.decode("utf-8")

                if data == 'in':
                    self.n_inputs+=1
                elif data == 'out':
                    self.n_outputs+=1
                else:
                    self.n_hidden+=1
                
                node = Node(_type=data)
                self.nodes.append(node)
            
            #getting innovation numbers#
            W = pd.read_csv( parent_file.open('innov.csv'))
            W = W.to_numpy()[:,1]
            innov = W.tolist()

            W = pd.read_csv( parent_file.open('in.csv'))
            W = W.to_numpy()[:,1]
            in_list = W.tolist()

            W = pd.read_csv( parent_file.open('out.csv'))
            W = W.to_numpy()[:,1]
            out_list = W.tolist()

            W = pd.read_csv( parent_file.open('weight.csv'))
            W = W.to_numpy()[:,1]
            weight= W.tolist()

            W = pd.read_csv( parent_file.open('enabled.csv'))
            W = W.to_numpy()[:,1]
            enabled= W.tolist()

            for i in range(0,len(innov),1):
                edge =Edge(innov[i],in_list[i],out_list[i],_weight=weight[i],_enabled=enabled[i])
                self.edges.append(edge)

            
if __name__=="__main__":

    model = Genome(n_inputs=2,n_outputs=1,init_pop=True)
    
    x = [1,1]

    print('before saving')
    model.save_model()
    print(model.feedforward(x))

    print('loading model')
    model.load_model()
    print(model.feedforward(x))


    #testing#
    # x = [1,1]
    # output = model.feedforward(x)
    # print(output)

