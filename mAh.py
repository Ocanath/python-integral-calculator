import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integrate

Beta = 3

t = np.linspace(0,5000,100000)
x = t**4*np.exp(-Beta*(t**2))

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
    return (t**4)*np.exp(-Beta*(t**2))

ix = num_integral(t, x, 0)
ixscipi = integrate.odeint(f, 0, t)

# solution_guess = (1/4)*np.sqrt(np.pi / (Beta**3)) 
solution_chatgpt = 3*np.sqrt(np.pi)/(8*Beta**(5/2))

# maxerr = np.max(np.abs(ix - (1/2)*x**2))
# print(maxerr)

print("my value: ", ix[len(ix)-1])
print("scipi value: ", ixscipi[len(ixscipi)-1])
print("guess value: ", solution_chatgpt)
fig,ax = plt.subplots()
ax.plot(t,ix)
ax.plot(t, ixscipi)
plt.show()
