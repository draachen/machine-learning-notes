### Logistic Regression

-   **Logistic regression**

    --> classifying data into discrete outcomes

    -   **Classification**

        --> use linear regression isn't a good idea

        logistic regression: $0<= h_\theta(x)<= 1$

    -   **Hypothesis Representation**

        $h_\theta(x) = g(\theta^Tx)= 1/(1 + e^{-\theta^Tx})$

        Sigmoid function, or Logistic function

        ![week3_sigmoid_function](.\resource\week3_sigmoid_function.png)

        Interpretation: $h_\theta(x)$=estimated probability that y = 1 on input x

        $P(y=0|x;\theta) = 1 - P(y=1|x;\theta)$

    -   **Decision boundary**

        predict $"y =  1"$ if $h_\theta(x) >= 0.5$

        predict $"y =  0"$ if $h_\theta(x) < 0.5$

    -   **Cost function**

        $Cost(h_\theta(x), y) = \cases{ -log(h_\theta(x)) & if y = 1 \cr -log(1-h_\theta(x)) & if y = 0 }$

        ![week3_logistic_regression_cost_function](.\resource\week3_logistic_regression_cost_function.png)


    -   **Simplified cost function and gradient descent**

        $$J(\theta) = \frac 1m [\sum_{i=1}^my^{(i)}\log h_\theta (x^{(i)})+ (1-y^{(i)})\log (1-h_\theta(x^{i}))]$$

        want $min_\theta J(\theta)$

        Repeat $ \{\\ \theta_j := \theta_j - \alpha \frac{\partial }{\partial \theta_j} J(\theta)\\ \}$

        ​	(simultaneously update all $\theta_j$)

    -   **Advanced optimization**

        Optimization algorithms:

        -   Gradient descent
        -   **Conjugate gradient**
        -   **BFGS**
        -   **L-BFGS**
    
        Advantages:
    
        -   No need to manually pick $\alpha$
        -   Often faster than gradient descent
    
        Disadvantages:
    
        -   More complex
    
    -   **Multi-class classification: One-vs-all**
    
        ![week3_one_vs_all](.\resource\week3_one_vs_all.png)
    
        Train a logistic regression classifier $h_\theta^{(i)}(x)$ for each class $i$ to predict the probability that $ y = i$

-   **Regularization**
    -   **The problem of overfitting**

        Overfitting: If we have too many features, the learned hypothesis may fit the training set very well, but fail to generalize to new examples.

        ![week3_overfitting](.\resource\week3_overfitting.png)

        Addressing overfitting:

        Options:

         - Reduce number of features.
           -   Manually select which features to keep.
           -   Model selection algorithm(later in course)
        - Regularization.
          - Keep all the features, but reduce magnitude/ values of parameter $\theta_j$
          - Works well when we have a lot of features, each of which contributes a bit to predicting $y$.

    -   **Cost function**

        $$J(\theta) =  \frac 1m [\sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})^2 + \lambda\sum_{j=1}^n \theta_j^2]$$

        $\lambda\sum_{j=1}^n \theta_j^2$  : regularization parameter

    -   **Regularized linear regression**

        cost function $J(\theta)$ changed

        gradient descent also changed: 

        $\theta_j := \theta_j(1 - \alpha \frac \lambda m) - \alpha \frac 1m \sum_{i=1}^m(h_\theta(x^{(i)}) - y^{(i)})x_j^{(i)}$

    -   **Regularized logistic regression**

        similar w/ regularized linear regression

### Programming Exercise 2: Logistic regression

-   Part i: Logistic Regression
    -   Visualizing the data
    -   Implementation
        -   Cost function and gradient
        -   Learning parameters using $fminunc$
        -   Evaluating logistic regression
-   Part ii: Regularized logistic regression
    -   Visualizing the data
    -   Feature mapping
    -   Cost function and gradient
        -   Learning parameters using $fminunc$
    -   Plotting the decision boundary

### Changelog

-   17.10.01 finish draft summary
-   17.09.24 add content
-   17.09.10 init create

Reference

-   [Andrea Ng: Machine Learning](https://www.coursera.org/learn/machine-learning)