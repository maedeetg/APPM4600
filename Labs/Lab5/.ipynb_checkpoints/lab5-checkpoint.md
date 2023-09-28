# Ecercises 3.1

1) Below is the adjusted Newton's method using fixed point iteration

`g(x_n) = x_n - \frac{f(x_n)}{f'(x_n)}` where `x_n` is a fixed point

Here are the necessary conditions for at least quadratic convergence:

* The function $f(x_n)$ must be twice continuously differentiable

* $f'(x_n) \neq 0$

* $\frac{f''(x_n)}{f'(x_n)}$ needs to be bounded i.e. $\frac{sup(f'')}{sup(f')} < M$

These conditions are true for a neighborhood around the fixed point. 


3) Yes, I needed to change the inputs, I added the derivative of the function as an input

5) The advantages of this new method is that we are finding a point that lies in the basin of convergence for Newton's method, so it is guarnteed to converge if the function that we are looking at follows the conditions from 1)

6) Bisection took 34 iterations, Newton's took 26 and the hybrid method took 34 iterations with a tolerance of $10^(-10)$. With a tolerance of $10^(-5)$, Bisection took 17 iterations, Newton's took 25 and the hybrid method took 17 iterations. 