# Lab 7

### Pre-lab

We have the points ${(x_j, f(x_j))}$ for for $j = 0, 1, ..., n$. 

We have the polynomial $p_{n}(x) = a_0 + a_1x + a_2x^2 + ... + a_nx^n$

To create the linear system in order to find the coefficients, we create `n+1` polynomials, where we plug in each $x_j$ into $p_{n}(x)$

So, we have

* $p_{n}(x_0) = a_0 + a_1x_0 + a_2x_0^2 + \dots + a_nx_0^n$

* $p_{n}(x_1) = a_0 + a_1x_1 + a_2x_1^2 + \dots + a_nx_1^n$

* $p_{n}(x_2) = a_0 + a_1x_2 + a_2x_2^2 + \dots + a_nx_2^n$

...

* $p_{n}(x_n) = a_0 + a_1x_n + a_2x_n^2 + \dots + a_nx_n^n$

We can write the linear system as

$$\begin{pmatrix} p_{n}x_0 \\ p_{n}x_1 \\ p_{n}x_2 \\ \vdots \\ p_{n}x_n \end{pmatrix} = \begin{pmatrix} 1 & x_0 & x_0^2 & \dots & x_0^n \\1 & x_1 & x_1^2 & \dots & x_1^n \\ 1 & x_2 & x_2^2 & \dots & x_2^n \\ \vdots & \vdots & \vdots & \dots & \vdots \\ 1 & x_n & x_n^2 & \dots & x_n^n \end{pmatrix} \begin{pmatrix} a_0 \\ a_1 \\ a_2 \\ \vdots \\ a_n \end{pmatrix}$$

Where $p_n = XA$

The code to solve for $A$ would be to solve for the inverse of $X$ and then complete the following matrix multiplication: $X^{-1}p_n = A$

### Outline of the rest of the lab

For the rest of the lab, we will be exploring interpolation. We showed in class that with each interpolation technique, we have a unique solution. i.e. there is one polynomial that we are searching for. 

The first step of the lab is to investivate different evaluation techniques. We will develop code to evaluate a polymomial using three different techniques

* Monomial expansion
* Lagrange polynomials
* Newton-Divided Differences

We will then evaluate polynomials of different degrees to see which methods perform the best and look at differences across the methods. 

After we investigate some prelimaries of each technique, we will improve our approximation of the interpolation polynomial through changing how we define our nodes. We will then observe the behavior of our polynomials with our new nodes compared to the previous method.
