import sigmoid.py as sig
import numpy
def cost(f) :
 f = numpy.array(f)
 data = []
 i = f.readline()
 while i :
  data.append(i)
  i = f.readline() 
 x=[]
 y=[]
 for i in data :
  x.append([1,i[0],i[1]])
  y.append([i[2]])
 x = numpy.array(x)
 y = numpy.array(y)
 c = numpy.array([0])
 for i in range(0,len(data)) :
  c = c + y[i]*numpy.log(sig.sigmoid(x[i].dot(transpose(f)))+(1-y[i])*numpy.log(1-sig.sigmoid(x[i].dot(transpose(f))))
 return c[0]/len(data)

def gradient(f) :
 f = numpy.array(f)
 data = []
 i = f.readline()
 while i :
  data.append(i)
  i = f.readline() 
 x=[]
 y=[]
 for i in data :
  x.append([1,i[0],i[1]])
  y.append([i[2]])
 x = numpy.array(x)
 y= numpy.array(y)
 g = numpy.array([0 for i in range (0,len(data))]) 
 for j in range(0,len(data)) :
  for i in range(0,len(data)) :
   g[j]= g[j] + (sig.sigmoid(x[i].dot(transpose(f)) - y[i])*x[i][j]
 g=g/len(data)
 return g

f=[1,4,3]
cost(f)
gradient(f)
