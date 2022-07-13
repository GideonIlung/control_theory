#IMPORTS#
import numpy as np
import time
import matplotlib.pyplot as plt
import turtle #will be used to show graphical simulation#


#GLOBAL PARAMS
TIMER = 10    
TIME_STEP = 0.005
SETPOINT = 10  #y value of where done should reach#
SIM_TIME = 1000  #duration of simulation#
INITIAL_X = 15
INTITIAL_Y = -100
MASS = 1 #kg
MAX_THRUST = 19.4 #Newtons
g = -9.81 #Gravitational constant
V_i = 0 #initial velocity 
Y_i = INTITIAL_Y #initial height
MAX_Y = 800
OFFSET = 20  #marker offset#
ANTIWINDUP = True
#-------------

#PID PARAMS
KP = 1
KI = 0.5
KD = 2
#----------------

class Simulation(object):
    """
        Creates graphical userface to view simulation
    """
    def __init__(self):
        #creating drone#
        self.drone  = Drone()

        #creating PID controller#
        self.pid = PID(KP,KI,KD,SETPOINT)

        #making marker#
        self.screen = turtle.Screen()
        self.screen.setup(1280,900)
        self.marker = turtle.Turtle()
        self.marker.penup() #lifts 'pen' so as object moves lines wont be made#
        self.marker.left(180) #rotates object by specified angle#
        self.marker.goto(INITIAL_X + OFFSET ,SETPOINT) #sets position#
        self.marker.color('red')
        self.sim = True
        self.timer = 0

        #for graphing#
        self.pos_list = []
        self.times = []
        self.error_list = []
        self.velo_list = []
        self.acc_list = []


    def cycle(self):

        while(self.sim):
            #getting thrust from PID controller#
            thrust = self.pid.compute(self.drone.get_postion())
            #-----------------------------------

            #updating drone values#
            self.drone.set_acceleration(thrust)
            self.drone.set_velocity()
            self.drone.set_position()
            
            time.sleep(TIME_STEP)
            self.timer+=1

            #getting values to plot#
            self.pos_list.append(self.drone.get_postion())
            self.velo_list.append(self.drone.get_velocity())
            self.acc_list.append(self.drone.get_acceleration())
            self.times.append(self.timer)
            self.error_list.append(self.pid.error)

            if self.timer > SIM_TIME:
                print('SIMULATION ENDED')
                self.sim = False
            elif self.drone.get_postion() > MAX_Y:
                print('OUT OF BOUNDS')
                self.sim = False
            elif self.drone.get_postion() < -MAX_Y:
                print('OUT OF BOUNDS')
                self.sim = False
        
        graph(self.times,self.pos_list,label = "postion",value_line=SETPOINT)
        #graph(self.times,self.velo_list,label = "velocity")
        #graph(self.times,self.acc_list,label = "acceleration")
        #graph(self.times,self.error_list,label = "error",value_line=0)

#PID controller#
class PID(object):
    def __init__(self,KP:float,KI:float,KD:float,target:float):
        self.kp = KP
        self.ki = KI
        self.kd = KD
        self.setpoint = target
        self.clamped = 0

        #initialising errors#
        self.error = 0

        #sums up the errors over time#
        self.integral_error = 0

        #derivative error looks at previous 2 errors#
        self.last_error = 0
        self.derivative_error = 0

        self.output = 0
    
    def compute(self,position:float)->float:
        """
            computes the errors of the the controller with respect to the 
            position of the drone

            position : the y value of the drone
        """
        #finding error#
        self.error = self.setpoint - position

        #integral error#
        #self.integral_error+= self.error * TIME_STEP   #intergral e(t)*dt#
        if (ANTIWINDUP==True) and (self.clamped!=0) and (((self.clamped< 0) and (self.error < 0)) or ((self.clamped>0)and (self.error>0))):
            self.integral_error+=0
        else:
            self.integral_error+= self.error * TIME_STEP   #intergral e(t)*dt#

        #derivative error#
        self.derivative_error = (self.error - self.last_error)/TIME_STEP

        #updating previous derivative error for next iteration#
        self.last_error = self.error

        #calculating output of the PID controller#
        self.output = (self.kp * self.error) + (self.ki * self.integral_error) + (self.kd * self.derivative_error)

        #clamping output to avoid saturation#

        #ANTIWINDUP#
        self.clamped = 0

        if self.output > MAX_THRUST:
            self.output = MAX_THRUST
            self.clamped = 1
        elif self.output < 0:
            self.output = 0
            self.clamped = -1
        
        return self.output


#DRONE OBJECT#
class Drone(object):
    def __init__(self):

        #creating object#
        global Drone #makes object a global variable#
        self.Drone = turtle.Turtle()
        self.Drone.penup()
        self.Drone.shape('square')
        self.Drone.goto(INITIAL_X,INTITIAL_Y)
        self.Drone.speed(0)

        self.ddy = 0 #acceleration#
        self.dy = V_i #velocity#
        self.y = Y_i #position#
    
    def set_acceleration(self,thrust:float):
        """
            sets the acceleration of the drone based on
            the thrust provided

            Fnett = ma; 
            Mass*g + thrust = Mass*a; 
            a = g + thrust/Mass

            Inputs:
                thrust : how much force is produced by rocket engine
        """
        self.ddy = g + thrust/MASS  
    
    def get_acceleration(self):
        """
            returns acceleration value
        """
        return self.ddy
    
    def set_velocity(self):
        """
            updates the velocity of the drone based on 
            the acceleration

            y'(t+1) = y'(t) + y''(t) 
        """
        self.dy+= self.ddy
    
    def get_velocity(self):
        """
            returns the velocity of the drone
        """
        return self.dy
    
    def set_position(self):
        """
            updates the position of the rocket
        """
        self.y+= self.dy
        self.Drone.sety(self.y)
    
    def get_postion(self):
        """
            returns drones y position
        """
        return self.Drone.ycor()

def graph(x,y,label,value_line):
    n = len(x)
    line = value_line * np.ones(n)
    plt.plot(x,y)
    plt.plot(x,line,'r--')
    plt.xlabel("time")
    plt.ylabel(label)
    plt.show()

#main function#
def main():
    sim = Simulation()
    sim.cycle()


#run main function#
main()
