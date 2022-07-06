#IMPORTS
import numpy as np
import zipfile
import os
import pandas as pd
import gym
import time
import copy
import matplotlib.pyplot as plt
from tikzplotlib import save as tikz_save

#number of iterations each chromosome trains#
TRAIN_TIME = 500

#number of iterations for simulation#
TIME = 500

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
            enabled  (bool)  : wether edge enabled or not
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
        self.update_fitness = 0
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
    def updated_fitness(self,n:int):
        """
            divides the fitness value of genome based
            on population of species

            Parameters:
                n (int) : population size of species genome belongs to
        """

        self.update_fitness = self.fitness/n

    def get_max_innov(self):
        """
            get the maximum innovation number in genome
        """

        innov = -1

        for x in self.edges:

            if x.innov > innov:
                innov = x.innov
        
        return innov
    
    def get_edge(self,innov:int):
        """
            gets the edge corresponding to innovation number
            if not found None is returned
        """

        for x in self.edges:
            if x.innov == innov:
                return x
        return None
    
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
    
    def add_edge(self,connections,count):

        I = self.n_inputs
        O = self.n_outputs
        H = self.n_hidden

        #no connections availabe#
        if len(connections) == 0:

            i = np.random.randint(0,I)
            j = np.random.randint(I,I+O)

            edge = Edge(count,i,j)
            self.edges.append(edge)
            return edge

        else:
            inputs = list(range(0,I,1)) + list(range(I+O,I+O+H,1))
            outputs = list(range(I,I+O,1)) + list(range(I+O,I+O+H,1))

            #maximum number of edges#
            max_edges = I*H + I*O + int((H*(H-1))/2) + H*O

            #no new edges to be added#
            if len(self.edges) == max_edges:
                return None

            #checking and searching new edges#
            i = np.random.choice(inputs)
            j = np.random.choice(outputs)
            exist = [k for k in range(0,len(self.edges),1) if (self.edges[k].in_node == i and self.edges[k].out_node ==j)]

            while (len(exist)!=0) or (i==j):
                i = np.random.choice(inputs)
                j = np.random.choice(outputs)
                exist = [k for k in range(0,len(self.edges),1) if self.edges[k].in_node == i and self.edges[k].out_node ==j]
            
            match = [k for k in range(0,len(connections),1) if connections[k].in_node == i and connections[k].out_node == j]
            
            if len(match)!=0:
                k = match[0]
                edge = Edge(connections[k].innov,i,j)
                self.edges.append(edge)
                return None
            else:
                edge = Edge(count,i,j)
                self.edges.append(edge)
                return edge
    
    def add_node(self,count):

        index = len(self.nodes)
        node = Node(_type="hid")
        self.nodes.append(node)

        self.n_hidden+=1

        i = np.random.randint(0,len(self.edges))
        self.edges[i].enabled = 0

        in_node = self.edges[i].in_node
        out_node = self.edges[i].out_node

        edge1 = Edge(count,in_node,index)
        edge2 = Edge(count+1,index,out_node)

        self.edges.append(edge1)
        self.edges.append(edge2)

        return edge1,edge2
    
    def mutate_weight(self,mu):

        change = False

        for x in self.edges:

            r = np.random.uniform(0,1)

            if r < mu:
                x.weight = x.weight + np.random.uniform(0,0.5)
                change = True
        
        return change

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

##############################QUICKSORT#############################################
def partition(x:list,left:int,right:int):

    pivot = x[right].update_fitness
    i = left - 1

    for j in range(left,right,1):
    
        if x[j].update_fitness >= pivot:
            i+=1
            temp_x = copy.deepcopy(x[i])
            x[i] = copy.deepcopy(x[j])
            x[j] = copy.deepcopy(temp_x)
    
    temp_x = copy.deepcopy(x[i+1])
    x[i+1] = copy.deepcopy(x[right])
    x[right] = copy.deepcopy(temp_x)

    return i+1

def q_sort(x:list,left:int,right:int):

    if left < right:
        part_index = partition(x, left, right)
        q_sort(x, left, part_index-1)
        q_sort(x, part_index+1, right)

def quickSort(x):
    n = len(x)
    q_sort(x,0,n-1)
####################################################################################   
###############################NEAT ALGORITHM ####################################################

def simu_fitness(x):
    """
        one run of the simulation

        Parameters:
            x (Genome) : the genome used to determine fitness
    """

    #add in simulation here#
    env_name = 'CartPole-v1'
    env = gym.make(env_name)

    state = env.reset()

    score = 0

    for count in range(TRAIN_TIME):
        action = x.feedforward(state)
        state,reward,done,info=env.step(int(np.round(action)))
        score+=reward

        if done == True:
            break
    
    env.close()
    return score

def fitness(x,rep=3):
    """
        determines the fitness of the genome

        Parameters:
            x   (Genome) : network to determine fitness on
            rep (int)    : number of simulation trials
    """

    total = 0

    for _ in range(0,rep,1):
        temp = simu_fitness(x)
        total+= temp
    
    total = total/rep
    x.fitness = total
    return total

def init_pop(n_inputs,n_outputs,m):
    pop = []
    costs = []

    connections = []

    max_edge = n_inputs * n_outputs

    #creating population#
    for _ in range(0,m,1):
        x = Genome(n_inputs=n_inputs,n_outputs=n_outputs,init_pop=True)

        no_edges = np.random.randint(1,max_edge)

        for _ in range(0,no_edges,1):
            edge = x.add_edge(connections,len(connections))
            if edge!=None:
                connections.append(edge)
        
        fx = fitness(x)

        pop.append(x)
        costs.append(fx)
    
    return pop,costs,connections

def compatability(x:Genome,y:Genome,count,c1=1,c2=1,c3=0.4):
    """
        determines the compatability between 2 genomes

        Parameters:
            x  (Genome) : first genome
            y  (Genome) : second genome
            c1 (float)  : excess edge parameter
            c2 (float)  : disjoint edge parameter
            c3 (float)  : average matching weight parameter
    """

    E = 0
    D = 0
    W = 0
    match = 0

    max_innov_x = x.get_max_innov()
    max_innov_y = y.get_max_innov()

    N = max(len(x.edges),len(y.edges))

    if N < 20:
        N = 1

    for i in range(0,count,1):

        temp1 = x.get_edge(i)
        temp2 = y.get_edge(i)

        #theres a match#
        if (temp1 !=None) and (temp2 !=None):
            average = (temp1.weight + temp2.weight)/2
            match+=1
            W+= average
        elif (temp1!=None) and (temp2==None) and (i < max_innov_y):
            D+=1
        elif (temp1==None) and (temp2!=None) and (i < max_innov_x):
            D+=1
        elif (temp1!=None) and (temp2==None) and (i > max_innov_y):
            E+=1
        elif (temp1==None) and (temp2!=None) and (i > max_innov_x):
            E+=1
    
    if match > 0:
        W = W/match

    delta = c1*(E/N) + c2*(D/N) + c3*W
    return delta

def speciation(pop,count,threshold):
    """
        performs speciation to determine genomes to be used for
        next generation
    """

    species = []

    #grouping genomes#
    for x in pop:

        if len(species) != 0:

            match = False

            for i in range(0,len(species),1):

                y = species[i][0]
                delta = compatability(x, y, count)

                if delta < threshold:
                    species[i].append(x)
                    match = True
                    break
            
            if match == False:
                new_species = []
                new_species.append(x)
                species.append(new_species)

        else:
            new_species = []
            new_species.append(x)
            species.append(new_species)
    
    #Computing updated fitness and sorting#
    for i in range(0,len(species),1):

        for j in range(0,len(species[i]),1):
            species[i][j].updated_fitness(len(species[i]))
        
        quickSort(species[i])
    

    new_pop = []
    for i in range(0,len(species),1):

        n = len(species[i])

        m = n//2
        # if m == 0:
        #     m = 1
    
        for j in range(0,m,1):
            new_pop.append(species[i][j])
    
    costs = []
    for x in new_pop:
        costs.append(x.fitness)
    
    return new_pop,costs

def crossover(x:Genome,y:Genome,connections:list):

    #parent fitness#
    fx = x.fitness
    fy = y.fitness

    #number of inputs#
    I = x.n_inputs

    #number of outputs
    O = x.n_outputs

    #nodes for new genome#
    nodes = []

    for _ in range(0,I,1):
        node = Node(_type="in")
        nodes.append(node)
    
    for _ in range(0,O,1):
        node = Node(_type="out")
        nodes.append(node)

    #edges of new genome
    edges = []

    for i in range(0,len(connections),1):

        temp1 = None
        temp2 = None
        
        u = connections[i].in_node
        v = connections[i].out_node

        x_match = [k for k in x.edges if (k.in_node == u) and (k.out_node == v)]
        y_match = [k for k in y.edges if (k.in_node == u) and (k.out_node == v)]

        if len(x_match) != 0:
            temp1 = copy.deepcopy(x_match[0])
        
        if len(y_match) != 0:
            temp2 = copy.deepcopy(y_match[0])
        
        #if both parents have edge#
        if (temp1 != None) and (temp2!=None):

            r = np.random.uniform(0,1)

            if r < 0.5:
                edges.append(temp1)
            else:
                edges.append(temp2)
        
        else:

            new_edge = None

            if fx == fy:

                if temp1 != None:
                    new_edge = temp1
                else:
                    new_edge = temp2

            elif fx > fy:
                new_edge = temp1
            elif fx < fy :
                new_edge = temp2     

            if new_edge != None:
                edges.append(new_edge)
    

    max_node = -1

    for x in edges:

        if x.in_node > max_node:
            max_node = x.in_node
        
        if x.out_node > max_node:
            max_node = x.out_node
    
    H = max_node + 1 - I - O

    for _ in range(0,H,1):
        node = Node(_type="hid")
        nodes.append(node)
    
    #creating child#
    child = Genome(n_inputs=I,n_outputs=O)
    child.n_hidden = H
    child.nodes = nodes
    child.edges = edges

    fchild = fitness(child)
    child.fitness = fchild
    return child,fchild

def NEAT(n_inputs,n_outputs,m,n,mu_link=0.4,mu_node=0.3,mu_edge=0.2,get_iter=False,learn_curve=False):
    """
        Algorithm that finds the optimal weights and structure of 
        Nueral Network

        Parameters:
            n_inputs  (int)   : number of inputs
            n_outputs (int)   : number of outputs
            m         (int)   : population size
            n         (int)   : number of generations
            mu_link   (float) : probability of adding edge
            mu_node   (float) : probability of adding node
            mu_edge   (float) : probability of mutating edge

    """

    #initialising population#
    pop,costs,connections = init_pop(n_inputs, n_outputs, m)
    best = [np.max(costs)]

    for count in range(0,n,1):
        new_pop,new_costs = speciation(pop,len(connections), threshold=1)

        #creating children via crossover#
        n_children = m - len(new_pop)

        for _ in range(0,n_children,1):

            i = 0
            j = 0

            while (i == j):
                i = np.random.randint(0,len(new_pop)) 
                j = np.random.randint(0,len(new_pop))
            
            p1 = new_pop[i]
            p2 = new_pop[j]

            x,fx = crossover(p1, p2, connections)
            new_pop.append(x)
            new_costs.append(fx)
        

        #mutating links#
        if mu_link > 0:
            for x in new_pop:

                change = False
                r = np.random.uniform(0,1)
    
                if r < mu_link:
                    edge = x.add_edge(connections,len(connections))
                    change = True
                    if edge != None:
                        connections.append(edge)
                
                if change == True:
                    fx = fitness(x)
                    x.fitness = fx

        #mutating nodes#
        if mu_node > 0:
            for x in new_pop:
                change = False
                r = np.random.uniform(0,1)

                if r < mu_node:
                    change = True
                    edge1,edge2 = x.add_node(len(connections))

                    if edge1!=None:
                        connections.append(edge1)
                    
                    if edge2!=None:
                        connections.append(edge2)
        
        if mu_edge > 0 :
            for x in new_pop:
                change = x.mutate_weight(mu_edge)
                if change == True:
                    fx = fitness(x)
                    x.fitness = fx

        pop = copy.deepcopy(new_pop)
        costs = []

        for x in pop:
            costs.append(x.fitness)
        
        best.append(np.max(costs))

        if (get_iter == True) and (np.max(costs) == 500):
            return None,count+1
    
    if get_iter == True:
        return None,n+1
    
    if learn_curve == True:
        return best

    index = np.argmax(costs)
    x = pop[index]
    fx = costs[index]
    return x,fx


def run(I,O,render):
    model = Genome(I,O)
    model.load_model()

    env_name = 'CartPole-v1'
    env = gym.make(env_name)
    state = env.reset()

    error = []

    for _ in range(TIME):
        
        #plotting#
        error.append(state[2])

        action = model.feedforward(state)
        state,reward,done,info=env.step(int(np.round(action)))

        if render == True:
            env.render()
            time.sleep(0.08)

        if done == True:
            state = env.reset()
    
    env.close()

    if render == False:
        plt.plot(error,label='error')
        plt.show()

def analysis(I, O,rep = 30):
    model = Genome(I,O)
    model.load_model()

    env_name = 'CartPole-v1'
    env = gym.make(env_name)

    init_state = np.array([0.01,0.01,0.01,0.01])
    errors = []
    
    mean = []
    std = []
    time_data = []

    for i in range(0,rep,1):
        state = env.reset()
        env.state = init_state
        state = env.state
        error = []

        for _ in range(TIME):
        
            #plotting#
            error.append(state[2])
            start = time.time()
            action = model.feedforward(state)
            end = time.time()
            time_data.append(end-start)
            state,reward,done,info=env.step(int(np.round(action)))

            if done == True:
                break
        
        errors.append(error)
    
    env.close()

    X = np.array(errors)

    M,N = X.shape

    for i in range(0,N,1):

        value = X[:,i].mean()
        std.append(X[:,i].std())
        mean.append(value)
    
    time_data = np.array(time_data)
    response_time = time_data.mean()*1000

    mean = np.array(mean)
    std = np.array(std)

    print("average response time: ",response_time, " milliseconds")
    print("average error ",np.abs(mean).mean())
    print('average std error' ,np.abs(std).std())

    t = np.arange(len(mean))
    plt.plot(mean,label='mean displacement')
    plt.fill_between(t,mean - std, mean + std, color='b', alpha=0.2)
    plt.ylabel(r'displacement $\theta$')
    plt.xlabel(r'time $t$')
    plt.legend(loc='best')
    tikz_save('NEAT_plot.tikz')
    plt.show()

def learn_rate(I,O,rep=30):
    
    output = []

    for _ in range(0,rep,1):
        _,temp = NEAT(I,O, m=20, n=30,get_iter=True)
        output.append(temp)

    d = {'values':output}
    data = pd.DataFrame(d)
    data.to_csv("NEAT_learn_rate.csv")
    print("mean: ",data['values'].mean())

def learning_curve(I,O,rep=30):
    
    data = []
    mean = []
    std = []

    for _ in range(0,rep,1):
        temp = NEAT(I,O, m=20, n=30,learn_curve=True)
        data.append(temp)
    
    X = np.array(data)

    for i in range(0,N,1):

        value = X[:,i].mean()
        std.append(X[:,i].std())
        mean.append(value)

    #######saving results to text file##################################
    resultsfile = open('NEAT_learn_curve.txt','w')

    lines = []

    #looping through mean values#
    string = str(mean[0])

    for i in range(1,len(mean),1):
        string = string + ',' + str(mean[i])
    
    string = string + '\n'
    resultsfile.writelines(string)

    #looping through std values#
    string = str(std[0])

    for i in range(1,len(std),1):
        string = string + ',' + str(std[i])
    string = string + '\n'
    resultsfile.writelines(string)
    resultsfile.writelines(str(response_time))
    resultsfile.close()
    ##################################################

    mean = np.array(mean)
    std = np.array(std)

    t = np.arange(len(mean))
    plt.plot(mean,label='learning curve')
    plt.fill_between(t,mean - std, mean + std, color='b', alpha=0.2)
    plt.ylabel('duration without constraint violation')
    plt.xlabel('iteration')
    plt.legend(loc='best')
    plt.show()


if __name__=="__main__":

    n_inputs = 4
    n_outputs = 1
    #population size#
    m = 20
    #number of generations#
    n = 30

    #OPTIMISATION#
    # x,fx = NEAT(n_inputs, n_outputs, m, n)
    # x.save_model()
    # x.print_genome()
    # print(fx)

    #run(n_inputs, n_outputs)
    #analysis(n_inputs, n_outputs,rep = 30)
    #run(n_inputs, n_outputs,render=True)

    learn_rate(n_inputs,n_outputs)
