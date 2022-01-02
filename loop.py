##A loop that implements a method like the ones in main_file(Needs to be renamed) and testImageNeurons. Then takes those results and sends them to the motors. 
# Actually sending them to the motors have not been fully implemented yet. But the file in KTH-Neuro-Computing-Systems/Multi-motor does this. So just connecting these two 
# should work.   


#0 - Loop function, in Projct methods

#0.5 Data received from cameras and put into a generator

#1- Data creation, continous data - Generator

#1.5 Sparse data to tensor usable by Norse

#2- Data used as input for Neurons

#2.5- Complex network to calculate movement. 

#3- Neurons result used to get angle

#3.5- Relative angle for motors calculated

#4- Angle sent to motors for movement

#4.5 Benchmark results saved for one loop through

#5 - Benchmark calculated ( Time taken, etc)

import ProjectMethods as pm
import torch
import numpy as np
import matplotlib.pyplot as plt
import norse
import random as rand
from norse.torch import lif_step, lif_feed_forward_step, lif_current_encoder, LIFParameters
from norse.torch import li_step, LICell, LIState, LIParameters, LIFCell
from norse.torch.module import leaky_integrator as li
import random
from norse.torch.functional import lif as lif
from time import perf_counter_ns as pc
from scipy.signal import convolve2d
from torch.nn import Conv2d as conv2


def nanosecond_to_milisecond(t):
    return t*1e-6

#0 - Loop function
def loopthrough(times):
   #Testloop

   loop1 = loop(100,100)

   data_gen = loop1.data_generator()

   #Tensor for holding benchmark times
   b_times = torch.zeros(times)



   for x in range(times):
      time_start = pc() #util.nanosecond_to_milisecond(pc())

      input_tensor = loop1.sparse_data_to_tensor(data_gen.__next__())
      loop1.input_to_neurons(input=input_tensor)

      #Calculating angle and sending it to motors
      #This could/should only be done every so often

      angle = loop1.calculate_angle()
      #loop1.angle_to_motors(angle)

      #Calculate and save benchmark time
      b_times[x] = nanosecond_to_milisecond(pc()-time_start)


      #print("Time to run one step = {} milliseconds".format(util.nanosecond_to_milisecond(pc()-time_start)))
      if x % 50 == 49:
         print("Angle sent to motors {}".format(angle))
         #loop1.angle_to_motors(angle)
   
   #Note, using CPU since lack of access to CUDA GPU
   print("Average time to run one timestep {} milliseconds".format(sum(b_times)/times))

class loop():

   #Initiate LIF neurons with parameter p2
   def initiate_neurons(self):
      p2 = LIFParameters(tau_syn_inv = torch.as_tensor(1.0 / 5e-3), 
      tau_mem_inv = torch.as_tensor(0.7 / 1e-2), v_leak = torch.as_tensor(0), 
      v_th = torch.as_tensor(1))
      self.states =  None
      self.cells = LIFCell(p=p2)


   #Initiation of loop
   def __init__(self, size_x, size_y, nr_of_neurons = 100):
      self.size_x = size_x # Size of input image in the x dimension
      self.size_y = size_y # Size of input image in the y dimension
      self.nr_of_neurons = nr_of_neurons # Nr of neurons used
      self.activity_position = pm.randPixel(size_x, size_y) #Position of simulated activity in the image
      self.max_loops = 10000 #Maximum number
      self.spikes = torch.zeros(nr_of_neurons)
      self.timestep = 0
      self.initiate_neurons()
      print("activity position = {}".format(self.activity_position))
      print("Which should have the motor angles at  {}, {}".format(1800+((self.activity_position[0]/100)*700), 1800+((self.activity_position[1]/100)*700)))


   #1- Data creation, continous data - Generator
   def data_generator(self, rand_act=1, tru_act=1):
      while self.timestep < self.max_loops:
         yield pm.get_one_t_array(self.size_x, self.size_y, self.activity_position, rand_act, tru_act)
         self.timestep += 1
         
         #Reduce spikes continously, or better yet. Have the spikes be in a sparse array themselves. 
         #if self.timestep %  = 9
   
   #1.5 Sparse data to tensor usable by Norse
   def sparse_data_to_tensor(self, list):
      array = torch.zeros(self.size_x, self.size_y)
      for val in list:
            array[val[0], val[1]] = 1

      kernel = torch.ones([10,10])
      convolved = convolve2d(array, kernel, mode="valid")
      array2 = torch.from_numpy(convolved[::10, ::10]).flatten()


      return array2

   #2- Data used as input for Neurons
   def input_to_neurons(self, input):
      
      spike_output, self.states = self.cells(input_tensor=input, state=self.states)
      self.spikes = self.spikes + spike_output
      return self.spikes, self.states

   #3- Neurons result used to get angle
   def calculate_angle(self, k=10):

      #Print spikes if you want
      #print("Spikes: {}".format(spikes))

      tp_val, tp_ind = torch.topk(self.spikes, k)
      #print(self.spikes.size())

      #Print spike indices if you want
      #print("Spike maximum indices: {}".format(tp_ind))

      
      #Spikes to avg position
      avg = torch.tensor([0,0])
      for x, nr in enumerate(tp_ind, start = 0):
         avg = avg + (pm.neuron_nr_to_coord(nr) * (tp_val[x]/sum(tp_val)))
      #avg = avg/tp_ind.size(0)


      """ if self.timestep % 50 == 49:
         for nr in tp_ind:
            print(nr)
            print(pm.neuron_nr_to_coord(nr)) """


      #Print spike Spike_max approximate position  
      #print("Spike_max approximate position : {}".format(avg/tp_ind.size(0)))
      
      motor_angles = torch.tensor([1800+((avg[0]/100)*700), 1800+((avg[1]/100)*700)])

      #Print motor angle
      #print("Which corresponds to motor_angle ")

      return motor_angles


   #4- Angle sent to motors for movement
   def angle_to_motors(self, angle):
      print("Angle {} sent to motor (TBF)".format(angle))


   #4.5 Benchmark results saved for one loop through
   #def benchmark():
      
""" 
loop1 = loop(100,100)

data_gen = loop1.data_generator()

#25 len, 20 for random, 5 for true
#print(data_gen.__next__()[5])


#Testloop
time_start = pc() #util.nanosecond_to_milisecond(pc())
list = data_gen.__next__()
input_tensor = loop1.sparse_data_to_tensor(list)
loop1.input_to_neurons(input=input_tensor)
angle = loop1.calculate_angle()
loop1.angle_to_motors(angle)
print("Time to run one step = {} milliseconds".format(util.nanosecond_to_milisecond(pc()-time_start)))
 """

if __name__ == "__main__":
   loopthrough(100)

   time = pc() 
   print(pc()- time)

