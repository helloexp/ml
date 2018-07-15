
import matplotlib.pyplot as plt

import numpy as np

x=np.arange(0,10)

y=2*x
z=-x+3

plt.figure()
plt.plot(x,y)
plt.plot(x,z)
plt.xlim(0,3) #set the x axis limition
plt.ylim(0,3)

plt.axvline(x=0,color='grey') # draw a vertical lines
plt.axhline(y=0,color='grey')

plt.show()
plt.close()





