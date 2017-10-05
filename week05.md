### Neural Networks Learning:

-   **Cost function**

    -   Neural Network(Classification)

        -   L = total no. layers in network

        -   $S_l$ = no. of units(not counting bias unit) in layer $l$

            | Binary classification | Multi-class classification(K class) |
            | --------------------- | ----------------------------------- |
            | $y$ = 0 or 1          | $y   \in \rm I\! R^K$               |
            | 1 output unit         | K output units                      |

        -   Cost function

            ![week4_neural_network_cost_function](.\resource\week4_neural_network_cost_function.png)

-   **Back propagation algorithm**

    -   min$J(\theta)$

    -   Need code to compute:

        -   $J(\theta)$
        -   $\frac{\partial }{\partial \theta_ij^{(l)}}J(\theta)$

    -   Gradient computation

        ![week4_neural_network_gradient_computation_1](.\resource\week4_neural_network_gradient_computation_1.png)

        ![week4_neural_network_gradient_computation_2](.\resource\week4_neural_network_gradient_computation_2.png)

    -   **Back propagation algorithm**

        ![week4_neural_network_back_propagation_algorithm](.\resource\week4_neural_network_back_propagation_algorithm.png)

-   **Back propagation intuition**

    -   Forward Propagation

    -   Back Propagation

        How well is the network doing on example i?

        ![week4_forward_back_propagation_intuition](.\resource\week4_forward_back_propagation_intuition.png)

-   **Implementation note: Unrolling parameters**

    -   thetaVec, DVec

        ```octave
        thetaVec = [Theta1(:); Theta2(:); Theta3(:)];
        DVec = [D1(:); D2(:); D3(:)];

        Theta1 = reshape(thetaVec(1:110), 10, 11);
        Theta2 = reshape(thetaVec(111:220), 10, 11);
        Theta3 = reshape(thetaVec(221:231), 1, 11);
        ```

    -   Learning Algorithm

        -   Having initial parameters $\Theta^{(1)}, \Theta^{(2)}, \Theta^{(3)}$.

        -   Unroll to get $initialTheta$ to pass to

            $fminunc(@costFunction, initialTheta, options)$

            $function [jval, gardientVec] = costFunction(thetaVec)$

            -   From $thetaVec$, get  $\Theta^{(1)}, \Theta^{(2)}, \Theta^{(3)}$.
            -   Use forward pro/back prop to compute  $D^{(1)}, D^{(2)}, D^{(3)}$.
            -   and unroll $D^{(1)}, D^{(2)}, D^{(3)}$ to get $gradientVec$

-   **Gradient checking**

    -   Numerical estimation of gradients

        -   Implement:

        for $\theta \in scalar$

        ~~~ octave
        gradApprox = (J(theta + EPSILON) - J(theta - EPSILON))/(2*EPSILON))
        ~~~

        for  $\theta   \in \rm I\! R^n$

        ![week4_gradient_checking_for_theta](.\resource\week4_gradient_checking_for_theta.png)

        -   Implementation Note:
            -   Implement back prop to compute DVec (unrolled $D^{(1)}, D^{(2)}, D^{(3)}$).
            -   Implement numerical gradient check to compute $gradApprox$.
            -   Make sure they give similar values.
            -   Turn off gradient checking. Using back prop code for learning.

-   **Random initialization**

    -   for neural networks, $initialTheta$ should be zeros;

    -   Random initialization: Symmetry breaking

        Initialize each $\Theta_ij^{(l)}$ to a random value in [$-\epsilon$, $\epsilon$]

        ~~~octave
        Theta1 = rand(10, 11) * (2 * INIT_EPSILON) - INIT_EPSILON;
        Theta2 = rand(1, 11) * (2 * INIT_EPSILON) - INIT_EPSILON;
        ~~~

-   **Putting it together**

    -   Training a neural network

        -   Pick a network architecture(connectivity pattern between neurons)
            -   No. of input units: Dimension of features $x^{(i)}$
            -   No. of output units: Number of classes
            -   Reasonable default: 1 hidden layer, or if > 1 hidden layer, have same no. of hidden units in every layer(usually the more the better)

    -   Randomly initialize weights

    -   Implement forward propagation to get $h_\Theta(x^{(i)})$ for any $x^{(i)}$

    -   Implement code to compute cost function $J(\Theta)$

    -   Implement back prop to compute partial derivatives $\frac{\partial }{\partial \theta_ij^{(l)}}J(\theta)$

    -   Use gradient check to compare $\frac{\partial }{\partial \theta_ij^{(l)}}J(\theta)$ computed using back propagation vs. suing numerical estimate of gradient of  $J(\Theta)$

        Then disable gradient checking code

    -   Use gradient descent or advanced optimization method with back propagation to try to minimize $J(\Theta)$ as a function of parameters $\Theta$

        ​

-   **Back propagation example: Autonomous driving(optional)**

###Programming Exercise 4: Neural Networks Learning

-   Neural Networks
    -   Visualizing the data
    -   Model representation
    -   Feedforward and cost function
    -   Regularized cost function
-   Back propagation
    -   Sigmoid gradient
    -   Random initialization
    -   Back propagation
    -   Gradient checking
    -   Regularized Neural Networks
    -   Learning parameters using $fmincg$
-   Visualizing the hidden layer
    -   Optional(ungraded) exercise

### Changelog

-   17.10.05 add summary contents
-   17.09.10 initial create template

Reference

-   [Andrea Ng: Machine Learning](https://www.coursera.org/learn/machine-learning)