# Prelab exercises

Both forward difference and backward difference converge linearly. We can see this through the error terms and they increase by around $10^{-1}$ at each iteration. 

# 3.2 Exercises

4) My slacker newton code performed better than my lab partner in terms of number of iterations, but from the condition I chose to recompute the Jacobian, my code never reevaluated the Jacobian, so it was just doing Lazy Newton.

# 3.3 Exercises

2) For a larger h_j, we need more iterations. In my case, I needed 3 iterations with $h_j = 10**{-7}|x_j|$ and needed 4 iterations with $h_j = 10**{-3}|x_j|$

# 3.4 Exercises

3) This method needs 7 iterations to converge to the root. This hybrid method is faster than the original Slacker Newton as it needed 10 iterations to converge. It is slower than 2d Newton with Jacobian approximations as both need around 4 iterations to converge. 