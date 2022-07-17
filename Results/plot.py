import matplotlib.pyplot as plt
import numpy as np
from tikzplotlib import save as tikz_save

if __name__=='__main__':
    #filenames = ['NE_GA_learn_curve.txt','NEAT_learn_curve.txt']
    #labels = ['NE (GA) mean learning curve','NEAT mean learning curve']
    #colours = ['b','r']

    filenames = ['NEAT_mean_learn_curve.txt','GA_mean_learn_curve.txt','PSO_mean_learn_curve.txt']
    labels = ['NEAT best learning curve','GA best learning curve','PSO best learning curve']
    colours = ['b','r','g']

    fig = plt.figure()
    ax = fig.add_subplot(111)

    for i in range(0,len(filenames),1):
        name = filenames[i]
        file1 = open(name,"r")

        data = file1.readlines()

        mean = list(map(float,data[0].split(',')))
        std = list(map(float,data[1].split(',')))

        mean = np.array(mean)
        std = np.array(std)

        # print("average response time for ",labels[i],': ',data[2], " milliseconds")
        # print("average error for ",labels[i],': ',np.abs(mean).mean())
        # print('average std error for ',labels[i],': ',np.abs(std).std())

        t = np.arange(len(mean))
        ax.plot(mean,label=labels[i])
        ax.fill_between(t,mean - std, mean + std, color=colours[i], alpha=0.2)

    ax.set_aspect('auto')
    #plt.ylabel(r'displacement $\theta$')
    plt.ylabel('number of iterations before constraint violation')
    #plt.xlim(0,500)
    plt.ylim(0,600)
    #plt.xlabel(r'time $t$')
    plt.xlabel('generations')
    plt.legend(loc='best')
    #tikz_save('NE_plot.tikz')
    plt.show()