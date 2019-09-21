from __future__ import print_function
from GTC import *

print("""
-------------------------------
Example from Appendix H2 of GUM
-------------------------------
""")

V = ureal(4.999,3.2E-3,independent=False)      # volt
I = ureal(19.661E-3,9.5E-6,independent=False)  # amp
phi = ureal(1.04446,7.5E-4,independent=False)  # radian

set_correlation(-0.36,V,I)
set_correlation(0.86,V,phi)
set_correlation(-0.65,I,phi)

R = result( V * cos(phi) / I )
X = result( V * sin(phi) / I )
Z = result( V / I )

print('R = {}'.format(R) )
print('X = {}'.format(X) )
print('Z = {}'.format(Z) )
print
print('Correlation between R and X = {:+.2G}'.format( get_correlation(R,X) ) )
print('Correlation between R and Z = {:+.2G}'.format( get_correlation(R,Z) ) )
print('Correlation between X and Z = {:+.2G}'.format( get_correlation(X,Z) ) )

print("""
(These are not exactly the same values reported in the GUM.
There is some numerical round-off error in the GUM's calculations.)
""")

