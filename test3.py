from sympy import series, Symbol
from sympy.functions import sin, cos, exp
from sympy.plotting import plot
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = 13,10
plt.rcParams['lines.linewidth'] = 2

# Define symbol
x = Symbol('x')

# Function for Taylor Series Expansion
def taylor(function, x0, n):
    """
    Parameter "function" is our function which we want to approximate
    "x0" is the point where to approximate
    "n" is the order of approximation
    """
    return function.series(x,x0,n).removeO()

print('sin(x) =', taylor(sin(x), 0, 4))
print('cos(x) =', taylor(cos(x), 0, 4))
print('e(x) =', taylor(exp(x), 0, 4))
print('sin(1) =', taylor(sin(x), 0, 4).subs(x,1))
print('cos(1) =', taylor(cos(x), 0, 4).subs(x,1))
print('e(1) =', taylor(exp(x), 0, 4).subs(x,1))

# This will plot sine and its Taylor approximations
p = plot(sin(x),taylor(sin(x),0,1),taylor(sin(x),0,3),taylor(sin(x),0,5),
         (x,-3.5,3.5),legend=True, show=False)

p[0].line_color = 'blue'
p[1].line_color = 'green'
p[2].line_color = 'firebrick'
p[3].line_color = 'black'
p.title = 'Taylor Series Expansion for Sine'
p.show()