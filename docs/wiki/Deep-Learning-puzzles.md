# Q1. You've following equation: y= 4x^2+1, and initial X value as X0=5, and learning rate n=0.01 then what will be the x1 for a Gradient Descent Algorithm?
Ans: In the context of the gradient descent algorithm, the goal is to iteratively update the input \( x \) in the direction opposite to the gradient of the function to minimize the function's value. The update rule is given by:

$$x_{\text{new}} = x_{\text{old}} - \eta \cdot \frac{dy}{dx}$$

Here:
- $x_{\text{new}}$ is the updated value of x,
- $x_{\text{old}}$ is the current value of x,
- $\eta$ is the learning rate,
- $\frac{dy}{dx}$ is the derivative of the function with respect to x.

For the given equation $y = 4x^2 + 1$ , let's find the derivative $\frac{dy}{dx}$:

$\frac{dy}{dx}$  = 8x \/4

Now, using the given values $x_{\text{old}} = 5$ and $\eta = 0.01 \$, we can calculate the updated value $x_{\text{new}}$:

$x_{\text{new}} = 5 - 0.01 \cdot 8 \cdot 5$

$x_{\text{new}} = 5 - 0.4$

$x_{\text{new}} = 4.6$

So, the updated value $x_1$ for the Gradient Descent Algorithm would be approximately 4.6.
