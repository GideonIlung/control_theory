import time
import numpy as np
import matplotlib.pyplot as plt

#PARAMS#
a = 0
b = 10
dx = 1
dt = 1
alpha = 1
beta = 1
D = 1
f = lambda x : 40 + np.sin(x)

#---------------#

#Control Params
K = 3
accuracy = 1 
external_heat = 5
bitlength = int(np.round(np.log2((2*external_heat)*(10**accuracy) + 1)))
TIME = 10
#---------------#

##############################QUICKSORT#############################################
def partition(x,fx,left:int,right:int):

    pivot = fx[right]
    i = left - 1

    for j in range(left,right,1):
    
        if fx[j] <= pivot:
            i+=1

            temp_x = x[i].copy()
            temp_f = fx[i]

            x[i] = x[j].copy()
            fx[i] = fx[j]

            x[j] = temp_x.copy()
            fx[j] = temp_f 
    
    temp_x = x[i+1].copy()
    temp_f = fx[i+1]

    x[i+1] = x[right].copy()
    fx[i+1] = fx[right]

    x[right] = temp_x.copy()
    fx[right] = temp_f

    return i+1

def q_sort(x,fx,left:int,right:int):

    if left < right:
        part_index = partition(x, fx, left, right)
        q_sort(x, fx, left, part_index-1)
        q_sort(x, fx, part_index+1, right)

def quickSort(x,fx):
    n = len(x)
    q_sort(x, fx,0,n-1)
#################################################################################### 

#######################GENETIC ALGORITHM######################################
def convert_chromo_to_matrix(x,k):

    n_x = int(np.round((b-a)/dx))
    u = []

    for i in range(0,len(x),bitlength):
        string = ""

        for j in range(0,bitlength,1):
            string = string + str(x[i+j])
        
        dvalue = int(string,2)
        val = -external_heat + (2*external_heat)*((dvalue)/(2**bitlength -1))
        u.append(val)

    u = np.array(u)
    u = np.reshape(u,(n_x+1,k))
    return u



def init_chromosome(L:int)->list:

    x = []

    for i in range(0,L,1):
        
        r = np.random.uniform(0,1)

        if r < 0.5:
            x.append(0)
        else:
            x.append(1)
    
    return x 

def init_population(N,L,w0,horizion):

    pop = []
    costs = []

    n_x = int(np.round((b-a)/dx))
    L = bitlength * (n_x +1) * horizion

    for _ in range(0,N,1):
        x = init_chromosome(L)
        fx = GA_fitness(w0, x, horizion) 

        pop.append(x)
        costs.append(fx)
    
    return pop,costs

def elitism(pop,costs,k):
    """
        returns the top k solutions to be used in the next generation

        Parameters :
            pop   (array) : the current population
            costs (array) : costs associated with each chromosome in population
    """
    
    x = pop.copy()
    fx = costs.copy()

    quickSort(x,fx) #sorting to determine#

    data = []
    fit = []

    for i in range(0,k,1):
        data.append(x[i])
        fit.append(fx[i])
    
    return data,fit


def crossover(u,v,w0,horizion,mode=1):
    """
        creates children using crossover

        Parameters:
            u       (list)  : the first parent
            v       (list)  : the second parent
            x0      (list)  : current state
            horizon (int)   : the prediction horizion
            mode    (int)   : single point crossover or double point crossover
    """

    L = len(u)
    x = u.copy()
    y = v.copy()

    #single point crossover#
    if mode == 1:
        i = np.random.randint(0,L-2)
        x[i:L] = v[i:L].copy()
        y[i:L] = u[i:L].copy()
    else: #double point crossover#
        i = np.random.randint(0,L-1)
        j = np.random.randint(0,L-1)

        a = min(i,j)
        b = max(i,j)

        x[a:b+1] = v[a:b+1].copy()
        y[a:b+1] = u[a:b+1].copy()
    
    fx = GA_fitness(w0, x, horizion)
    fy = GA_fitness(w0, y, horizion)

    return [x,y],[fx,fy]

def mutation(x0,u,mu,horizion):
    """
        given probability of mu, the solution with mutate

        Parameters:
            x0      (list)  : the current state
            u       (list)  : possible solution
            mu      (float) : probability of mutation
            penalty (bool)  : penality added to function or not, if not penality violation values set to inf
    """

    x = u.copy()

    change = False

    for i in range(0,len(x),1):
        r = np.random.uniform(0,1)

        if (r<mu) and (x[i]==10):
            x[i] = -10
            change = True
        elif (r<mu) and (x[i]==-10):
            x[i] = 10
            change = True
    
    fx = GA_fitness(x0, x, horizion)
    return change,x,fx


def new_pop(w0,pop,costs,mu,k,horizion):
    """
        creates the solutions to be used in the next generation

        Parameters:
            x0      (list)  : current state
            pop     (array) : the current population
            costs   (array) : costs associated with each chromosome in population
            mu      (float) : probability of mutation
            k       (int)   : no of elite solutions to be handed over to next gen
            penalty (bool)  : penality added to function or not, if not penality violation values set to inf
    """

    #getting elite solutions#
    x,fx = elitism(pop, costs, k)

    #repeatedly create children#
    while len(x) < len(pop):

        #selecting parents using tournament selection#
        parents = []
        for _ in range(0,2,1):
            i = 0
            j = 0 

            while (i == j) and (len(pop)>1):
                i = np.random.randint(0,len(pop)-1)
                j = np.random.randint(0,len(pop)-1)
            
            if costs[i]< costs[j]:
                parents.append(pop[i])
            else:
                parents.append(pop[j])
        

        #creating children using crossover#
        child,f_child = crossover(parents[0],parents[1], w0,horizion)

        x = x + child
        fx = fx + f_child
    
    #mutation#
    for i in range(0,len(x),1):
        change,temp,temp_f = mutation(w0, x[i], mu,horizion)

        if change == True:
            x[i] = temp.copy()
            fx[i] = temp_f
    
    #if x larger than pop size remove worst solutions#
    quickSort(x, fx)
    while len(x) > len(pop):
        del x[-1]
        del fx[-1]
    
    #updating population#
    return x,fx

def GA(w0,horizion,N,k,m,mu):
    """
        Parameters:
            w0       (list) : the current state
            N        (int)  : population size
            horizon  (int)  : prediction horizion
            k        (int)  : number of elite solutions to be kept
            m        (int)  : number of generations
            mu       (float): probability of mutation
    """

    #external heat source intensity range from -5 to 5#
    n_x = int(np.round((b-a)/dx))
    L = bitlength * (n_x +1) * k

    #initialising population#
    pop,costs = init_population(N, L, w0,horizion)

    for _ in range(0,m,1):
        pop,costs = new_pop(w0, pop, costs, mu, k, horizion)

    action = convert_chromo_to_matrix(pop[0], horizion)
    return costs[0],action


#######################SYSTEM FUNCTIONS########################################

def evaluate(w0,u):

    m,n = u.shape

    X = np.copy(w0)
    w = np.copy(w0)

    for i in range(0,n,1):
        wnew = get_state(w, u[:,i])
        X = np.hstack((X,wnew))
        w = np.copy(wnew)
    
    value = get_cost(X,n)
    return value

def GA_fitness(w0,x,k):
    """
        Determines a fitness of a chromosome

        Parameters:
            w0 (array) : the current state of the system
            x  (array) : the chromosome in question
            k  (int)   : the prediction horizion
    """
    u = convert_chromo_to_matrix(x, k)
    value = evaluate(w0, u)
    return value

def get_cost(X,k):
    """
        Returns the fitness of the states over a time horizion 

        Parameters:
            X (list) : the states 
            k        : prediction horizion
    """
    m,n = X.shape

    total = 0

    for i in range(0,m,1):
        for j in range(0,n,1):
            total += (X[i,j]-27)**2

    return total

def get_state(w,u):
    """
        Determines next state of system from current state and control 

        Parameters:
            w  (list) : the current state
            u  (list) : the control    
    """

    n_x = int(np.round((b-a)/dx))
    L = D*(dt/(dx**2))

    A = np.zeros((n_x+1,n_x+1))
    d = np.zeros((n_x+1,1))

    #setting first and last values#
    A[0,0] = 1-2*L
    A[0,1] = 2*L
    A[-1,-1] = 1-2*L
    A[-1,-2] = 2*L

    d[0] = -2*L*alpha*dx 
    d[-1] = 2*L*beta*dx

    #getting matrix values#
    for i in range(1,n_x,1):
        A[i,i-1] = L
        A[i,i] = 1-2*L
        A[i,i+1] = L 
    
    wnew = (A @ w) + d + u
    state = np.copy(wnew[:,0])
    state = np.reshape(state,(len(state),1))
    return state

def init_state():
    n_x = int(np.round((b-a)/dx))

    w = []

    for i in range(-1,n_x,1):
        x_i = a + (i+1)*dx
        w.append(f(x_i))

    w = np.array(w)
    w = w.astype('float64')
    w= np.reshape(w,(len(w),1))
    return w

def simulate(horizion):

    w = init_state()

    X = np.copy(w)

    for _ in range(0,TIME,1):
        cost,u = GA(w, horizion = horizion, N=20, k=10, m=50, mu=1e-3)
        wnew = get_state(w, u[:,0])

        #updating states#
        w = np.copy(wnew)
        X = np.hstack((X,w))
    
    plot_solution(X)
    

def plot_solution(states):
    fig = plt.figure()
    plt.rcParams['figure.figsize'] = (16,10)
    ax = plt.axes(projection = '3d')

    n_x = int(np.round((b-a)/dx))
    n_t = int(np.round((TIME)/dt))

    t_space = np.linspace(0,TIME,n_t+1)
    x_space = np.linspace(a,b,n_x+1)

    print(states.shape)
    print(t_space.shape)
    print(x_space.shape)

    X,T = np.meshgrid(t_space,x_space)

    ax.plot_surface(T,X,states,cmap='plasma',label='approx')
    plt.show()


if __name__ == '__main__':
    w0 = init_state()
    # # X = np.hstack((w0,w0))
    # # print(X)

    #cost,action = GA(w0, horizion = 3, N=20, k=10, m=50, mu=1e-3)

    # # k = 4
    # # n_x = int(np.round((b-a)/dx))

    # # L = bitlength * (n_x +1) * k
    # # x = init_chromosome(L)
    # # u = convert_chromo_to_matrix(x, k)

    # # value = evaluate(w0, u)

    # # print("initial state: \n",w0)
    # # print("\n control: \n",u)
    # # print("\nfitness: \n",value)

    # print("cost :",cost)
    # print("\n action: \n",action)
    simulate(horizion=3)

