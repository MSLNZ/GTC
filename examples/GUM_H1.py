from GTC import *

print("""
-------------------------------
Example from Appendix H1 of GUM
-------------------------------
""")

# Lengths are in nm
d0 = ureal(215,5.8,24,label='d0')  
d1 = ureal(0.0,3.9,5,label='d1')  
d2 = ureal(0.0,6.7,8,label='d2')

# Intermediate quantity 'd'
d = d0 + d1 + d2

alpha_s = ureal(11.5E-6, type_b.uniform(2E-6),label='alpha_s')
d_alpha = ureal(0.0, type_b.uniform(1E-6), 50,label='d_alpha')
d_theta = ureal(0.0, type_b.uniform(0.05), 2,label='d_theta')

theta_bar = ureal(-0.1,0.2,label='theta_bar')
Delta = ureal(0.0, type_b.arcsine(0.5),label='Delta')

# Intermediate quantity 'theta'
theta = theta_bar + Delta

l_s = ureal(5.0000623E7,25,18,label='l_s')   

# two more intermediate steps
tmp1 = l_s * d_alpha * theta
tmp2 = l_s * alpha_s * d_theta

# Final equation for the measurement result
l = result( l_s + d - (tmp1 + tmp2), label='l')

print( "Measurement result for l={}".format(l) )

print("""
Components of uncertainty in l (nm)
-----------------------------------""")

for i in reporting.budget(l):
    print( "  {!s}: {:G}".format(i.label,i.u) )