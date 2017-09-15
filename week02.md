### Multiple features 

-   **Multiple features(variables)**

    **Notation:**

     -   n = number of features
     -   $x^{(i)}$ = input (features) of $i^{th}$ training example
     -   $x^{(i)}_j$ = value of feature $j$ in $i^{th}$ training example

    **Hypotheses** for **Multivariate linear regression**
    $$
    h_\theta(x) = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + ... + \theta_n x_n
    $$

    $$
    --> h_\theta(x) = \theta^T x
    $$

    ​	Remark: for convenience of notation, define $x_0$ = 1

    ​	$x$ is (n+1) x 1 vector, $\theta$ is also (n+1) x 1 vector

-   **Gradient descent for multiple variables**

    ![week2_gradient_descent_for_multivariate_linear_regression](resource\week2_gradient_descent_for_multivariate_linear_regression.png)

-   **Gradient descent in practice**

    **Feature Scaling** 

    Idea: Make sure features are on a similar scale

    Target: Get every feature into approximately a $-1\leq x_i \leq 1$ range

    **Mean normalization**

    Remark: do not apply to $x_0$ = 1

    **Learning rate**

    -   If $\alpha$ is too small; slow convergence.
    -   If $\alpha$ too large; $J(\theta)$ may not decrease on every iteration; may not converge; slow convergence also possible.

-   **Features and polynomial regression**

    $h_\theta(x) = \theta_0 + \theta_1(size) + \theta_x(size)^2$

-   **Normal Equation**

    method to solve for $\theta$ analytically

    ![week2_normal_equation](resource\week2_normal_equation.png)

    $m$ examples $(x^{(1)}, y^{(1)}),...(x^{(m)}, y^{(m)})$; $n$ features


    Octave: $pinv (X'*X)*X'*y$

    ![week2_gradient_descent_and_normal_equation](resource\week2_gradient_descent_and_normal_equation.png)

-   **Normal equation and non-invertibility(option)**

    non-invertible usually caused by:

    -   Redundant feaures(linearly dependent)
    -   Too many features($e.g. m\leq n$)
        -   delete some features, or use regularization


### Octave tutorial

-   **Basic operations**
-   Moving data around
-   Computing on data
-   Plotting data
-   Control statement: for, while, if statements
-   Vectorial implementation
    -   can easily been solved in coding environments

### Programming Exercise 1: Linear Regression

-   Part i: linear regression with one variable

    ex1.m

    -   load data

    ~~~octave
    data = load('ex1data1.txt')  
    X = data(:, 1); y = data(:, 2); %load data into variable X and y
    m = length(y); %number of training example
    ~~~

    -   plot data

    ~~~octave
      '''
      function plotData(x, y)
      figure; 
      plot(x, y, 'rx', MarkerSize', 10);
      ylabel('Profit in $10,000s');
      xlabel('Population of City in 10,000s');
      end
      '''
    ployData(X, y) 
    ~~~

    -   cost

    ~~~octave
    X = [ones(m, 1), data(:,1)]; %add a column of ones to x
    theta = zeros(2,1); %initialize fitting parameter
    iterations = 1500;
    alpha = 0.01;
    ~~~

    -   computing the cost $J(\theta)$

    ~~~octave
      '''
      function J = computeCost(X, y, theta)
      m = length(y);
      J = 1/(2*m) * sum((X * theta-y).^2);
      end
      '''
    J = computeCost(X, y, theta)
    ~~~

    -   Gradient descent

    ~~~octave
      '''
      function theta = gradientDescent(X, y, theta, alpha, num_iters)
      m = length(y);
      J_history = zeros(num_iters, 1);
      for iter = 1:num_iters
          a = 1/msum(X(:,1)'(X*theta-y));
      	  b = 1/m*sum(X(:,2)'*(X*theta-y));
          delta = [a;b];
          theta = theta - alpha*delta;
          J_history(iter) = computeCost(X, y, theta);
      end
      
      end 
      '''
    theta = gradientDescent(X, y, theta, alpha, iterations); %run gradient descent
    ~~~

    -   Plot the linear fit

    ~~~octave
    hold on; % keep previous plot visible
    plot(X(:,2), X*theta, '-')
    legend('Training data', 'Linear regression')
    hold off % don't overlay any more plots on this figure
    ~~~

    -   Predict values for population sizes of 35,000 

    ~~~octave
    predict1 = [1, 3.5] *theta;
    fprintf('For population = 35,000, we predict a profit of %f\n',...
        predict1*10000);
    ~~~

    -   Visualize $J(\theta)$

    ~~~octave
    % Grid over which we will calculate J
    theta0_vals = linspace(-10, 10, 100);
    theta1_vals = linspace(-1, 4, 100);

    % initialize J_vals to a matrix of 0's
    J_vals = zeros(length(theta0_vals), length(theta1_vals));

    % Fill out J_vals
    for i = 1:length(theta0_vals)
        for j = 1:length(theta1_vals)
    	  t = [theta0_vals(i); theta1_vals(j)];
    	  J_vals(i,j) = computeCost(X, y, t);
        end
    end

    % Because of the way meshgrids work in the surf command, we need to
    % transpose J_vals before calling surf, or else the axes will be flipped
    J_vals = J_vals';
    % Surface plot
    figure;
    surf(theta0_vals, theta1_vals, J_vals)
    xlabel('\theta_0'); ylabel('\theta_1');

    % Contour plot
    figure;
    % Plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
    contour(theta0_vals, theta1_vals, J_vals, logspace(-2, 3, 20))
    xlabel('\theta_0'); ylabel('\theta_1');
    hold on;
    plot(theta(1), theta(2), 'rx', 'MarkerSize', 10, 'LineWidth', 2);
    ~~~


-   Part ii: linear regression with multiple variables

    n.a.

    ​

### Changelog 

-   17.09.10 init create @draachen

Reference

-   [Andrea Ng: Machine Learning](https://www.coursera.org/learn/machine-learning)
-   [Octave documentation pages](http://www.gnu.org/software/octave/doc/interpreter/)