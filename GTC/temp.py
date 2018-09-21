from GTC import *
import math 

x = ureal(10E7*math.pi,10)
print x

x = ureal(math.pi,1)
print x

x = ureal(10E7*math.pi,0.1)
print x

x = ureal(10E-6*math.pi,0.0001)
print x

C = 1
x = C * math.pi
y = C * math.pi
z = ucomplex( complex(x,y), 10 )
print z

C = 1
x = C * math.pi
y = C * math.pi
z = ucomplex( complex(x,y), 1)
print z