### simple linear modeling (regression)
Suppose we have some true value $y$ which we want to approximate in terms of $x$. 
- That is, an input of $x$ gives an output of $y$.
We guess that this relationship is linear. Therefore, our *estimate* for $y$ is "y hat":
$$
\hat{y}=mx+b
$$
If we write $y$ in terms of $\hat{y}$, it'd be this:
$$
y=\hat{y}+\epsilon
$$
where $\epsilon$ is called **irreducible error**.
> [!info] In general, $\hat{y}$ will not be a perfect estimate for $y$, thus introducing error that cannot be reduced.

--

We have multiple inputs $[x_{1},x_{2},\dotsc,x_{n}]$ and thus multiple outputs $[\hat{y}_{1},\hat{y}_{2},\dotsc,\hat{y}_{n}]$.
$$
\hat{y}_{i}=mx_{i}+b,y_{i}=\hat{y}_{i}+\epsilon_{i}
$$
So assuming our model is imperfect, our predictions $\hat{y}_{i}$ for $y$ will be off sometimes (or all the time). We want to measure how "off" we are so we can build a better prediction model. Define the **residual** for a single data point as $e_i = y_i - \hat{y}_i$.

### Minimizing the Mean Squared Error
Define the **mean squared error** (MSE) as
$$
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_{i})^2
$$
Think of the MSE as just a function. Open it up and see what variables you can change.
$$
\text{MSE}=\mathcal L(m,b)=\frac{1}{n}\sum^n_{{i=1}}(y_{i}-mx_{i}-b)^2
$$
So we have a function, it has two variables. In single-variable calculus you were taught that to find the maximum (or minimum) of a quadratic, you take the derivative of $y$ with respect to $x$ and set it equal to 0, then solve for the $x$ that maximizes the $y$. Similarly, here we take the *partial derivative* of the function $\mathcal L(m,b)$ with respect to $m$ and $b$. That's two partial derivatives.

However, we will <u>not</u> set the resolved partial derivatives to zero here and solve for the exact parameters. For a simple linear regression model with just two variables, you *could*. But for a real-world model with 175 billion parameters (e.g. early ChatGPT), it is computationally intractable. This is discussed in the extra reading.

**Gradient descent** is an iterative shortcut. Instead of doing one massive calculation, it takes small, cheap steps toward the bottom, which scales well to large datasets.

But first we need to calculate the partial derivatives.

#### Partial derivative with respect to $m$
Differentiate $\mathcal L(m, b)$ with respect to $m$, treating $b$ as a constant:
$$\frac{\partial J}{\partial m} = \frac{1}{n} \sum_{i=1}^{n} \frac{\partial}{\partial m} (y_i - (mx_i + b))^2$$
Applying the chain rule (bring the exponent to the front, then multiply by the derivative of the inside with respect to $m$, which is $-x_i$):
$$\frac{\partial \mathcal L}{\partial m} = \frac{1}{n} \sum_{i=1}^{n} 2(y_i - (mx_i + b)) \cdot (-x_i)$$
Clean it up by pulling the constants and the negative sign to the front:
$$\frac{\partial \mathcal L}{\partial m} = -\frac{2}{n} \sum_{i=1}^{n} x_i (y_i - \hat{y}_i)$$
#### Partial derivative with respect to $b$
Differentiate $\mathcal L(m, b)$ with respect to $b$, treating $m$ as a constant. The derivative of the inside with respect to $b$ is simply $-1$:
$$\frac{\partial \mathcal L}{\partial b} = \frac{1}{n} \sum_{i=1}^{n} 2(y_i - (mx_i + b)) \cdot (-1)$$
Cleaning up:
$$\frac{\partial \mathcal L}{\partial b} = -\frac{2}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)$$
#### Gradient Descent Update Rule
$$\theta_{\text{i+1}} = \theta_{\text{i}} - \alpha \cdot \nabla J(\theta)$$
Where $\alpha$ (alpha) is the **learning rate**, a small positive number controlling the *step size*.

For the weight/slope ($m$) we get:
$$m := m - \alpha \left( -\frac{2}{n} \sum_{i=1}^{n} x_i (y_i - \hat{y}_i) \right)$$
$$m := m + \frac{2\alpha}{n} \sum_{i=1}^{n} x_i (y_i - \hat{y}_i)$$
For the bias/intercept ($b$):
$$b := b - \alpha \left( -\frac{2}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i) \right)$$
$$b := b + \frac{2\alpha}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)$$


Extra (not relevant for neural networks)

### Extra reading: Least Squares and Normal Equations 
Define the **residual sum of squares** (RSS) as
$$
\text{RSS}=\sum^n_{i=1}\epsilon_{i}^2=\sum^n_{i=1}(y_{i}-\hat{y}_{i})^2
$$
We want to find the values of $m$ and $b$ that make this total sum as small as possible. 

If you take the partial derivative with respect to $b$, set it to 0, and solve for $b$, you get the following:
$$
b=\frac{1}{n}\sum^n_{i=1}y_{i}-m\cdot\frac{1}{n}\sum^n_{i=1}x_{i}
$$
> Guarantees that the best fit line passes through the *center of mass* of the data, $(\bar{x}, \bar{y})$.

Similarly, if you take the partial derivative with respect to $m$, set it to $0$, and solve for $m$ you get:
$$m = \frac{\sum_{i=1}^{n} (x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^{n} (x_i - \bar{x})^2}$$
> The optimal slope is the sample covariance of $x$ and $y$ divided by the sample variance of $x$.

This solution is called **Least Squares**. The above equations are the scalar versions of what are called the **Normal Equations**. Resolving the above takes a lot of algebra. But when you write them in terms of matrices (when you have more than just two parameters $m$ and $b$, but a general weight vector $\vec \theta$ it's a lot easier and a shorter derivation. The final result looks cleaner too:
$$
\vec \theta=(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^{T}\mathbf{y}
$$
The Normal Equations give you the exact answer immediately: it's matrix multiplication. However, calculating an inverse matrix is computationally expensive (goes as $\mathcal O(n^3)$), and this cost becomes intractable for neural networks that underlie LLMs.

Also, the normal equations only work when your model is linear. E.g. works for multiple linear regression, but not nonlinear regression. Neural networks are inherently nonlinear, and there are no general methods available for solving nonlinear systems. Most attempts at solving nonlinear systems are based on iterative schemes, such as gradient descent.

> [!notes] More reading
> [Notes by Alex Townsend](https://math.mit.edu/icg/resources/teaching/18.085-spring2015/LeastSquares.pdf)
> [Notes by David Bindel](https://www.cs.cornell.edu/~bindel/class/cs3220-s12/notes/lec10.pdf)

> There's also the deal with convex and non-convex functions. Typical neural networks are not convex, due to their nonlinear activations as well some other details (hidden layers). We'll talk about this later.