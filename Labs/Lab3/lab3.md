# Problem 1

a) For the interval $(a, b) = (0.5, 2)$, bisection methods gives us a good root estimate. The error message reads `0`, so this is a success. It estimates `root = 0.9999999701976776` and `f(astar) = -2.98023206113385e-08`. We get a good estimate as the interval given has $f(a)$ and $f(b)$ differ sign and it crosses over the x-axis.

b) For the interval $(a, b) = (-1, 0.5)$, bisection method yeilds a failure. It estimates `root = -1` and `f{astar) = -2`. Bisection does not work for this interval for the function as $f(a)$ and $f(b)$ are the same sign, so the function does not cross the x-axis. Because of this, the function will set the estimate to $a$ and exit. 

c) For the interval $(a, b) = (-1, 2)$, bisection method gives us a good root estimate. The error message reads `0`, so this is a success. It estimates `root =  0.9999999701976776` and `f(astar) = -2.98023206113385e-08`. This estimate is good, but not as good as the estimate in part a). We get a good estimate as the interval given has $f(a)$ and $f(b)$ differ sign and it crosses over the x-axis.

**Question:** Is it possible for bisection to find the root `x = 0` for this function?

**Answer:** No, the root at $x = 0$ does not have a change in sign, i.e. it does not cross the x-axis, so bisection will never detect it. The only way it could is by guessing it. If we had an interval $(a, b) = (0, 0.5)$, the interval would not change sign, but the algorithm would guess the root to be $a$, and then exit. 


# Problem 2

a) `f(x) = (x - 1)(x - 3)(x - 5)`. 

For the interval $(a, b) = (0, 2.4)$, bisection method gives us a good root estimate. The error message reads `0`, so this is a success. It estimates `root = 1.0000030517578122` and `f(astar) = 2.4414006618542327e-05`. From how the function is written, we can see that there is a root at $x = 1$ and $f(a)*f(b) < 0$, so with bisection method and our interval, we will be able to estimate the root at $x = 1$. 

b) `f(x) = (x - 1)^2(x - 3)`

For the interval $(a, b) = (0, 2)$, bisection method yeilds a failure. i.e. the error message reads `1`. It estimates `root = 0` and `f(astar) = -3`. Bisection does not work for this function as it is attempting to find a root of multiplicity 2. Bisection method will only work for roots that have a multiplicity equal to 1. 

c) `f(x) = sin(x)`

For the interval $(a, b) = (0, 0.1)$, bisection method gives us a good root estimate. The error message reads `0`, so this is a success. It estimates `root = 0` and `f(astar) = 0.0`. $sin(x)$ has a root at the origin $x = 0$. This interval does not actually have a sign change, it only yeilds the correct estimate of the root since the left endpoint in the interval, $a$, happens to be the root. 

For the interval $(a, b) = (0.5, 3*np.pi / 4)$, bisection method yeilds a failure. i.e. the error message reads `1`. It estimates `root = 0.5` and `f(astar) = 0.479425538604203`. Bisection does not work for this function as the interval does not contain a root. There is no sign change as well, so it estimates the roots as the left endpoint, $0.5$, and exits the algorithm. 

# Problem 3

a) `f(x) = x(1 + {7 - x^5}/{x^2}^3`

With `fixedpt = 7^(1/5)` for this function, the fixed point algorithm does not converge. This algorithm does not converge as it can be seen that within 2 iterations, it blows up. With the given starting point, after the first iteration of fixed point algo, it will be past the fixed point it is trying to find, so it will continue to try and find the fixed point, but it never will since it already passed it. That is why it blows up so quickly. 

b) `f(x) = x - {x^5 - 7}/{x^2}`

Same reasoning from part a)

c) `f(x) = x - {x^5 - 7}/{5x^4}`

For this function, the fixed point algorithm converges. With the max number of iterations being 500, it can estimate the fixed point exactly. This is because it does not pass the actual fixed point after the first iteration. It will just get closer and closer to the fixed point after each iteration. 

d) `f(x) = x - {x^5 - 7}/{12}`

Same reasoning from part c)
