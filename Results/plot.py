import matplotlib.pyplot as plt
import numpy as np
import tikzplotlib
#from tikzplotlib import save as tikz_save

def single_plot():
    filenames = ['PID/PID_noise_results_1.txt','PID/PID_noise_results_2.txt','PID/PID_noise_results_3.txt']
    labels = [r'$\dagger$',r'$\diamond$',r'$\ast$']
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

        #print("average response time for ",labels[i],': ',data[3]*1000, " milliseconds")
        print("average error for ",labels[i],': ',np.abs(mean).mean())
        print('average std error for ',labels[i],': ',np.abs(std).std())

        t = np.arange(len(mean))
        ax.plot(mean,label=labels[i])
        ax.fill_between(t,mean - std, mean + std, color=colours[i], alpha=0.2)

    ax.set_aspect('auto')
    plt.ylabel(r'displacement $\theta$')
    #plt.ylabel('number of iterations before constraint violation')
    #plt.xlim(0,500)
    #plt.ylim(0,600)
    plt.xlabel(r'time $t$')
    #plt.xlabel('generations')
    plt.legend(loc='best')
    tikzplotlib.save("PID_noise_results.tex",axis_height='10cm',axis_width='16cm')
    #tikz_save('test_plot.tikz',axis_height='\\figH',axis_width='\\figW')
    plt.show()

def double_plot():
    filenames1 = ['PID/PID_results_1.txt','PID/PID_results_2.txt','PID/PID_results_3.txt']
    labels1 = [r'$\dagger$',r'$\diamond$',r'$\ast$']
    filenames2 = ['PID/PID_noise_results_1.txt','PID/PID_noise_results_2.txt','PID/PID_noise_results_3.txt']
    labels2 = [r'$\dagger$',r'$\diamond$',r'$\ast$']
    colours = ['b','r','g']

    fig,(ax1,ax2) = plt.subplots(2,1,sharex=True)

    for i in range(0,len(filenames1),1):
        name = filenames1[i]
        file1 = open(name,"r")

        data = file1.readlines()

        mean = list(map(float,data[0].split(',')))
        std = list(map(float,data[1].split(',')))

        mean = np.array(mean)
        std = np.array(std)

        #print("average response time for ",labels[i],': ',data[3]*1000, " milliseconds")
        print("average error for ",labels1[i],': ',np.abs(mean).mean())
        print('average std error for ',labels1[i],': ',np.abs(std).std())

        t = np.arange(len(mean))
        ax1.plot(mean,label=labels1[i])
        ax1.fill_between(t,mean - std, mean + std, color=colours[i], alpha=0.2)
        ax1.set_title('Ideal Enviroment')
        ax1.set_ylabel(r'displacement $\theta$')
        ax1.legend(loc='best')

    for i in range(0,len(filenames2),1):
        name = filenames2[i]
        file1 = open(name,"r")

        data = file1.readlines()

        mean = list(map(float,data[0].split(',')))
        std = list(map(float,data[1].split(',')))

        mean = np.array(mean)
        std = np.array(std)

        #print("average response time for ",labels[i],': ',data[3]*1000, " milliseconds")
        print("average error for ",labels2[i],': ',np.abs(mean).mean())
        print('average std error for ',labels2[i],': ',np.abs(std).std())

        t = np.arange(len(mean))
        ax2.plot(mean,label=labels2[i])
        ax2.fill_between(t,mean - std, mean + std, color=colours[i], alpha=0.2)
        ax2.set_title('Noisy Enviroment')
        ax2.legend(loc='best')
        ax2.set_ylabel(r'displacement $\theta$')

     
    #plt.ylabel(r'displacement $\theta$')
    #plt.ylabel('number of iterations before constraint violation')
    #plt.xlim(0,500)
    #plt.ylim(0,600)
    plt.xlabel(r'time $t$')
    #plt.xlabel('generations')
    #plt.legend(loc='best')
    tikzplotlib.save("PID_results.tex",axis_height='10cm',axis_width='16cm')
    #tikz_save('test_plot.tikz',axis_height='\\figH',axis_width='\\figW')
    plt.show()
if __name__=='__main__':
    #single_plot()
    double_plot()
    
