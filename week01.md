### Introduction

- Examples:

  - Database mining
  - Applications can't programm by hand  
  - Self-customizing programs  
  - Understanding human learning

- Definition

  > A computer program is said to learn from experience E with respect to some task T and some performance measure P, if its performance on T, as measured by P, improves with experience E.    by Tom Mitchell (1998)

- Algorithm:
  - supervised learning
  - unsupervised learning
  - Others:
  - reinforcement learning
  - recommender systems

- Supervised learning: use dataset to predict
  - Regression: continuous value
  - Classification: discrete value  

- Unsupervised learning:
  - clustering  
  - non-clustering  

   for examples: social net analysis; market segmentation; astronomical data analysis

### Linear regression with one variable
- Model representation

| Supervised Learning                      | Regression Problem           |
| ---------------------------------------- | ---------------------------- |
| - Given the 'right answer' for each example in the data | - Predict real-valued output |

![structure](resource\week1_machine_learning_structure.png)



​	Hypothesis: $h_\theta (x) = \theta_0 + \theta_1 x$

-   Notations:  
    -   m: Number of training examples  
    -   x's: 'input' variable/features  
    -   y's: 'output' variable/ 'target' variable  
    -   (x,y): single training example  
    -   ($x^{i}, y^{i}$): the $i^{th}$ training example  

-   Cost function

    **Hypothesis:** $$h_\theta (x) = \theta_0 + \theta_1 x$$

    **Idea:**  Choose $\theta_0, \theta_1$ so that $h_\theta (x)$ is close to y for our training example(x, y) 

    **Parameters:** $\theta_0, \theta_1$

    **Cost function:** 

    also called squared error function, used to measure the accuracy of hypotheses function
    $$
    J(\theta_0, \theta_1) = \frac{1}{2m}\sum_{i=1}^m(h_\theta(x^{(i)})-y^{i})^2
    $$
    **Goal:** $ minimize \quad J(\theta_0, \theta_1)$

    1.  <u>**Simplified:**</u>

    $$
    h_\theta(x) = \theta_1x
    $$

    $$
    J(\theta_1) = \frac{1}{2m}\sum_{i=1}^m(h_\theta(x^{(i)}-y^{(i)})^2
    $$

    $$
    minimize\quad J(\theta_1)
    $$




![week1_h(x)_and_J(theta1)](resource\week1_h(x)_and_J(theta1).png)

       	2.  w/ $\theta_0, \theta_1$

![week1_h(x)_and_J(theta0, theta1)](resource\week1_h(x)_and_J(theta0, theta1).png)

-   **Gradient descent**

    **Outline:** 

     - Start w/ some $\theta_0, \theta_1$
    - Keep changing $\theta_0, \theta_1$ to reduce $J(\theta_0, \theta_1)$ until we hopefully end up at a minimum

![week1_gradient_descent_J(theta0,theta1)](resource\week1_gradient_descent_J(theta0,theta1).png)

​	**Gradient descent algorithm**

​	repeat until convergence {
$$
\theta_j := \theta_j -\alpha \frac{ \partial}{ \partial \theta_j}J(\theta_0, \theta_1)\quad (for j = 0\:and\: j = 1)
$$
​	}

​	Remark: Simultaneous update $\theta_0, \theta_1$

​	**Learning rate $\alpha$ definition**

![week1_gradient_descent_alpha](resource\week1_gradient_descent_alpha.png)

​	Remark: Gradient descent can converge to a local minimum, even with the learning rate $\alpha$ fixed.

-   **Gradient descent for linear regression**

![week1_gradient_descent_and_linear_regression](resource\week1_gradient_descent_and_linear_regression.png)

​	repeat until convergence {
$$
\theta_0 := \theta_0 -\alpha \frac{1}{ m}\sum_{i=1}^m(h_\theta(x^{i})-y^{i})
$$

$$
\theta_1 := \theta_1 -\alpha \frac{1}{ m}\sum_{i=1}^m(h_\theta(x^{i})-y^{i})x^{i}
$$

​	}

​	**"Batch" Gradient Descent:** 

​	Each step of gradient descent uses all the training examples

### Linear algebra review

-   Matrices and vectors

    Dimension of matrix: number of rows x number of columns

    Matrix Elements: entries of matrix

    ​	$A_{ij}$ = "i, j entry" in the $i^{th}$ row and $j^{th}$ column

    Vector: An n x 1 matrix, called 'n-dimensional vector'

-   Addition and scalar multiplication

-   Matrix-vector multiplication

    ​	A		X 		$x$ 		= 		$y$

    To get $y^{i}$, multiply A's $i^{th}$ row with elements of vector $x$, and add them up

-   Matrix-matrix multiplication

    ​	A		X 		B 		= 		C

    The $i^{th}$ column of the matrix C is obtained by multiplying A with the $i^{th}$ column of B. (for $i$ = 1,2,3,...,o)

-   Matrix multiplication properties

    A x B $\neq$ B x A 	(not commutative, AB not same dimension)

    Identity Matrix $I​$

    ​	For any matrix A,

    ​	A x $I$ = $I$ x A = A		(two $I$ may not same dimension)

-   Inverse and transpose

    **Matrix inverse:**

    ​	A should be a square matrix, and if it has an inverse,

    ​	AA$^{-1}$= A$^{-1}$A = $I$

    **Matrix Transpose:**

    ​	$B$ = $A^T$		means $B_{ij}=A_{ji}$



### Changelog

-   17.09.10 change formate from .ipyn to .md
-   17.09.09  init create @draachen 

Reference:

-   [Andrea Ng: Machine Learning](https://www.coursera.org/learn/machine-learning)