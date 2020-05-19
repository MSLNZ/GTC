from __future__ import print_function
from GTC import *

print("""
-------------------------------
Example of using new functions:
pinv - pseudoinverse matrix
online pseudoinverse matrix calculator: https://comnuan.com/cmnn0100f/
-------------------------------
""")

x = la.uarray([[2, 1], [3, 4], [1, 2]])
x_pinv = la.pinv(x)
print('matrix value:',x)
print('pseudoinverse matrix value:',x_pinv)
print("""
-------------------------------
Example of using new functions:
psolve - Function solves overdetermined (more equations than variables) equation system with using Moore–Penrose pseudoinverse matrix
-------------------------------
""")
a = la.uarray([[2, 1], [-3, 1], [-1, 1]])
b = la.uarray([[4], [-1], [0.98]])
# x should be close [[1],[2]]
x=la.psolve(a, b)
print('x=',x)
