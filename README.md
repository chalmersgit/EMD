Created on 11/04/2015

@author: Andrew Chalmers

This code computes the Earth Mover's Distance, as explained here:
http://homepages.inf.ed.ac.uk/rbf/CVonline/LOCAL_COPIES/RUBNER/emd.htm

This is done using numpy, scipy (minimize)

There is a simple example of two distributions computed by getExampleSignatures()
This example is chosen in order to compare the result with a C implementation 
found here:
http://robotics.stanford.edu/~rubner/emd/default.htm