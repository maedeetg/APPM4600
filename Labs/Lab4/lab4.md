### 3.2 Exercises

We can see from the Aitken's code that this method converges faster to the fixed point than the fixed point iteration. From the example in prelab, with the fixed point iteration, it took 12 iterations to converge. With Aitkens, it took 4.

Also, another example to show that it converges much faster for a larger number of iterations, I used the function $f(x) = x - \frac{x^5 - 7}{12}$ with the starting point $x0 = 1.0$ and a tolerance of $10e-10$. With the fixed point iteration, we can see that in about 836 iterations, it converges to the fixed point value, where Aikten's converges in 220. Wow!! 