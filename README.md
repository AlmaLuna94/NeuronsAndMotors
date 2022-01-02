# NeuronsAndMotors

This repository contains 3 main files that do similar things: testImageNeurons, locate_activity and loop. As well as file containing extra functions that are used in all three previous files.

Locate_activity and testImageNeurons are two quite similar methods to locate where in a noisy spiking data image the main activity is happening.
Loop also tries to locate activity in a noisy spiking data image, but it also implements many steps to be able to send that location data to motors that can follow the activity in real time. It also loops through many times to see how it can follow a changing activity location over time, and times how long each iteration takes. 

To be able to actually move motors using this, all that is needed is to connect this code to the 
motor controlling code in the folder KTH-Neuro-Computing-Systems, specifically the code in motorControlPython to actually move it. But this has yet to be done. 

The spiking image data that these functions/files currently used is manually created to imitate DVS camera data. But with a function that continuously send data from a DVS camera into these functions, they should work fine without much if any further changes to the actual location finding methods. 

---------------------------------------------------------------------------------------------------

These functions/files use artificial neurons and functions for these neurons, from the library norse. At the moment no learning is done on the network of neurons and their connections. But since it is implemented using norse, which has ways of doing machine learning using these neurons. This could be a possible next step. 


---------------------------------------------------------------------------------------------------
The file loop can be run normally, with the value in loopthrough(x) representing how many timesteps are looped through.

locate activity is a notebook file which can be run through stepwise.

testImageNeurons is a notebook file as well which can be run through stepwise. You can change the values of T,N and U to change the size/length of the spiking input data. 
