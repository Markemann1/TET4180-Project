"""
author: @emilaunejakobsen
Assignment 4 - 1a
Plot of steady state power vs angle characteristic

"""
import matplotlib.pyplot as plt
import numpy as np

# Parameters
Eq = 1.47
Vth = 0.898
xd = 1.05
xq = 0.66

# Declaring coordinates
x = np.arange(0, 180, 1)

# Pe1, Pe2 and Pe
Pe1 = ((Eq * Vth) / xd) * np.sin(np.deg2rad(x))
Pe2 = ((Vth**2 * (xd-xq))/(2*xd*xq))*np.sin(np.deg2rad(2*x))
Pe = Pe1 + Pe2

# Pm
Pm = np.full((180,), 0.8)  # an array with 180 elements of 0.8, Pm = 0.8

# Axis limits
plt.ylim(-0.5, 2)
plt.xlim(0, 180)

# Axis labels
plt.ylabel('Power')
plt.xlabel('Angle')

# Plot title
plt.title('Steady state power vs angle characteristic - EAJ')

# Plot
plt.plot(x, Pe1)
plt.plot(x, Pe2)
plt.plot(x, Pe)
plt.plot(x, Pm)


"""plt.legend(['Pe1', 'Pe2', 'Pe', 'Pm'])
plt.show()"""
