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

# Systematic offset error
E_sys = ureal(0,0.005)  # the value of uncertainty is arbitrary
b_sys = [b_i + E_sys for b_i in b]

# Arbitrary reference temperature (degrees C)
t_0 = 20.0

# Calculate the temperature relative to t_0
t_rel = [ t_k - t_0 for t_k in t ]

# Least-squares regression for type-B
cal_b = tb.line_fit(t_rel,b_sys)
print( cal_b )
print

# Least-squares regression for type-A
cal_a = ta.line_fit(t_rel,b_sys)
print( cal_a )
print

# Combine results
intercept = ta.merge(cal_a.intercept,cal_b.intercept)
slope = ta.merge(cal_a.slope,cal_b.slope)

print( repr(intercept) )
print( repr(slope) )
print

# Apply correction at 30 C
b_30 = intercept + slope*(30.0 - t_0)

print("Correction at 30 C: {}".format(b_30))
