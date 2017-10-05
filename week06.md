### Advice for applying machine learning

-   **Deciding what to try next**

    -   Debugging a learning algorithm:

        | Actions                                  | Used to           |
        | ---------------------------------------- | ----------------- |
        | Get more training examples               | fix high variance |
        | Try smaller sets of features             | fix high variance |
        | Try getting additional features          | fix high bias     |
        | Try adding polynomial features ( $x_1^2, x_2^2, x_1x_2,$ etc) | fix high bias     |
        | Try decreasing $\lambda$                 | fix high bias     |
        | Try increasing $\lambda$                 | fix high variance |

-   **Evaluating a hypothesis**

    -   Dataset: 70% as training set, 30% as test set
    -   Model selection:
        -   for linear regression:
            -   minimize $J_{train}(\theta)$, compute $J_{test}(\theta)$
        -   for logistic regression:
            -   minimize $J_{train}(\theta)$, compute Misclassification error(0/1 misclassification error)

-   **Model selection and training/validation/test sets**

    -   Dataset: 60% as training set, 20% as cross validation set, 20% as test set
    -   Model selection:
        -   minimize $J(\theta^{(i)})$, then use $\theta^{(i)}$ to compute $J_{cv}(\theta^{(i)})$
        -   min$J_{cv}(\theta^{(i)})$ will be the selected parameters

-   **Diagnosing bias vs. variance**

    -   Bias/variance effected by degree of polynomial

        ![week5_bias_variance_polynomial](.\resource\week5_bias_variance_polynomial.png)

-   **Regularization and bias/variance**

    -   choosing the regularization parameter $\lambda$

        -   minimize $J(\theta^{(i)})$, then use $\theta^{(i)}$ to compute $J_{cv}(\theta^{(i)})$
        -   min$J_{cv}(\theta^{(i)})$ will be the selected parameters
        -   normally, $\lambda \in [0, 0.01, 0.02, 0.04, 0.08, ..., 10.24, etc]$ 

    -   Bias/variance as a function of the regularization parameter $\lambda$

        ![week5_bias_variance_regularization_lambda](.\resource\week5_bias_variance_regularization_lambda.png)

-   **Learning curves**

    -   High bias

        ![week5_learning_curve_high_bias](.\resource\week5_learning_curve_high_bias.png)

    -   High variance

        ![week5_learning_curve_high_variance](.\resource\week5_learning_curve_high_variance.png)

-   **Deciding what to try next(revisited)**

    -   Debugging a learning algorithm(see chart above)
    -   Neural networks and over fitting
        -   'Small' neural network: 
            -   Computationally cheaper
        -   'Large' neural network: 
            -   Computationally more expensive
            -   more prone to over fitting --> use regularization($\lambda$) to address over fitting

### Machine learning system design

-   **Prioritizing what to work on: Spam classification example**

    -   Building a spam classifier
        -   $x$ = features of e-mail;  $y $ = spam(1) or not spam(0)
        -   How to spend your time to make it have low error?
            -   Collect lots of data
            -   features based on e-mail routing information(from e-mial header)
            -   features about "discount" and "discounts", or "deal" and "Dealer"
            -   sophisticated algorithm to detect misspellings
            -   ...

-   **Error analysis**

    -   **Recommended approach**
        -   Start w/ a simple algorithm, implement and test on your cv set
        -   plot learning curves to decide if more data, more features, etc are likely to help
        -   Error analysis: manually examine wrong estimated examples
    -   The importance of numerical evaluation

-   **Error metrics for skewed classes**

    -   Skewed classes: e.g. only 0.5% patient have cancer

    -   Precision/Recall

        ![week6_precision_recall](.\resource\week6_precision_recall.png)

-   **Trading off precision and recall**

    -   Trading off precision and recall

        ![week6_trading_off_precision_recall](.\resource\week6_trading_off_precision_recall.png)

    -   **$F_1Score$ (F Score)**

        -   $$
            F_1Score = 2\frac{PR}{P+R}
            $$

-   **Data for machine learning**

    -   **Design a high accuracy learning system**

        >   "It's not who has the best algorithm that wins. It's who has the most data."

    -   **Large data rationale**

        -   enough features

        -   enough sophisticated algorithm

            --> low bias

        -   a very large training set(unlikely to over fit)

            --> low variance

### Programming Exercise 5: Regularized Linear Regression and Bias v.s. Variance

-    Regularized Linear Regression
    -   Visualizing the dataset
    -   Regularized linear regression cost function
    -   Regularized linear regression gradient
    -   Fitting linear regression
-   Bias-variance
    -   Learning curves
-   Polynomial regression
    -   Learning Polynomial Regression
    -   Optional(ungraded) exercise: Adjusting the regularization parameter
    -   Selecting $\lambda$ using a cross validation set
    -   Optional(ungraded) exercise: Computing test set error
    -   Optional(ungraded) exercise: Plotting learning curves with randomly selected examples

### Changelog

-   17.10.05 add summary content
-   17.09.10 initial create template

Reference

-   [Andrea Ng: Machine Learning](https://www.coursera.org/learn/machine-learning)