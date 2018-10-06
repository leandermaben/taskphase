import numpy
def sigmoid(n) :
  n = numpy.array(n)
  z = 1/(1+numpy.exp(-n))
  return z


