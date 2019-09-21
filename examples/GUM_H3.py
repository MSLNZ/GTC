from __future__ import print_function
from GTC import *

print("""
-------------------------------
Example from Appendix H3 of GUM
-------------------------------
""")
# Thermometer readings (degrees C)
t = (21.521,22.012,22.512,23.003,23.507,23.999,24.513,25.002,25.503,26.010,26.511)

# Observed differences with calibration standard (degrees C)
b = (-0.171,-0.169,-0.166,-0.159,-0.164,-0.165,-0.156,-0.157,-0.159,-0.161,-0.160)

# Arbitrary offset temperature (degrees C)
t_0 = 20.0

# Calculate the temperature relative to t_0
t_rel = [ t_k - t_0 for t_k in t ]

# Least-squares regression
cal = type_a.line_fit(t_rel,b)
print( cal )
print

# Apply correction at 30 C
b_30 = cal.intercept + cal.slope*(30.0 - t_0)

print("Correction at 30 C: {}".format(b_30))
