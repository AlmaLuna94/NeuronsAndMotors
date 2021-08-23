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


# a generator that yields a list of activities each time it is called
   #8 sum_of_first_n = sum(firstn(1000000))

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



   #1- Data creation, continous data - Generator
   def data_generator(self, rand_act=1, tru_act=1):
      while self.timestep < self.max_loops:
         yield (pm.get_one_t_array(self.size_x, self.size_y, self.activity_position, rand_act, tru_act), num)
         self.timestep += 1
         
         #Reduce spikes continously, or better yet. Have the spikes be in a sparse array themselves. 
         #if self.timestep %  = 9
   
   #1.5 Sparse data to tensor usable by Norse
   def sparse_data_to_tensor(self):
      array = torch.zeros(self.size_x, self.size_y)
      for val in array:
            array[val[0], val[1]] = 1
      return array

   #2- Data used as input for Neurons
   def input_to_neurons(self, input):
      
      spike_output, self.states = self.cells(input_tensor=input, state=self.states)
      self.spikes = self.spikes + spike_output
      return self.spikes, self.states

   #3- Neurons result used to get angle
   def calculate_angle():
      


   print("Spikes: {}".format(spikes))
tp_val, tp_ind = torch.topk(spikes, 4)
print("Spike maximum indices: {}".format(tp_ind))
avg = torch.tensor([0,0])
for nr in tp_ind:
    avg = avg + pm.neuron_nr_to_coord(nr)
print("Spike_max approximate position : {}".format(avg/tp_ind.size(0)))
pm.plotNeurons(voltages.detach(),N)



loop1 = loop(100,100)
data_gen = loop1.data_generator()
#25 len, 20 for random, 5 for true
#print(len(data_gen.__next__()[0]))


