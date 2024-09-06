import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate



t = np.linspace(-1,5,1000)
x = t

##initial value
def num_integral(t,x,x0):
    assert(len(t) == len(x))    #make sure the lengths match
    integral_of_x = x*0
    integral_of_x[0] = x0
    sum = x0 #gotta start somewhere amiright
    prev_t = t[0]
    for i in range(1,len(t)):
        sum = sum + (x[i]*(t[i] - prev_t))
        prev_t = t[i]
        integral_of_x[i] = sum
    return integral_of_x

def f(x, t):
    return x

ix = num_integral(t, x, 0.5)
ixscipi = integrate.odeint(f, 0, t)

maxerr = np.max(np.abs(ix - (1/2)*x**2))
print(maxerr)

fig,ax = plt.subplots()
# ax.plot(t,x)
ax.plot(t,ix)
ax.plot(t,(1/2)*x**2)
plt.show()
