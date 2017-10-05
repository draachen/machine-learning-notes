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

    $\theta = (X^TX)^{-1}X^Ty$


    Octave: $pinv (X'*X)*X'*y$


​	![week2_gradient_descent_and_normal_equation](.\resource\week2_gradient_descent_and_normal_equation.png)


-   **Normal equation and non-invertibility(option)**

    non-invertible usually caused by:

    -   Redundant features (linearly dependent)
    -   Too many features($e.g. m\leq n$)
        -   delete some features, or use regularization


### Octave tutorial

-   **Basic operations/ Moving data around**

    ~~~octave
    1~=2 % not equal
    1 && 0 %AND
    1 || 0 %or
    xor(0,1) %
    a = 3
    v = [1 2 3] %row vector
    v = [1;2;3] %column vector
    v = 1:0.1:2 %create a column vector, [1;1.1;1.2;...2]
    ~~~

    ~~~octave
    help ones %display the help text for ones function
    ones(2,3) %create a 2x3 matrixs, elements equals to 1
    rand(3,3) %create a random 3x3 matrixs, elements less than 1
    w = -6 + sqrt(10)*(rand(1,10))
    hist(w)
    hist(w, 5) %plot a histogram of the values in 5 bins
    eye(4) %return a identity matrix
    zeros(10,1) %return a 10x1 matrix whose elements are all 0

    size(A) %return the number of rows and columns of A
    size(X, 1) %return row of X
    size(x, 2) %return column of X

    [nr, nc] = size(A);
    length(A) %for empty object:0, for scalars:1, for vectors: elements qty
    		%for matrix: rows or columns qty, depends on which is larger
    V + ones(length(V), 1);

    data = load('featuresX.dat ');
    X = data(:, 1:3);
    y = data(:, 4);
    m = length(y); 

    A(2,:) %return 2nd row of matrix A
    A(:,2) %return 2nd column of matrix A
    A(:) %put all elements of A into a single vector
    A(:, 3:5) = B(:,1:3) %replace columns 3 to 5 of A with columns 1 to 3 of B
    A(:, 2: end) %return elements from 2nd to last column of A

    C = [A A] %column doubled
    C = [A,A] %column doubled
    C = [A;A] %row doubled
    A.*B %multiply corresponding elements of A,B
    abs(A) %absolute elements

    %max
    max(A) %return max element of matrix A
    [val, ind] = max(A, [], 1) %obtain max value of each column and its index, row vector format
    [val, ind] = max(A, [], 2) %obtain max value of each row and its index, column vector format

    pinv(A) % calculate inverse of matric 
    A' %calculate transpose of matrix A

    %Logical arrays
    a = 1:10; b = 3; % create a vector a, and a scalar b
    a == b; %return a vector same size of a, with ones at positions wjere the elements of a are equal to b, and zeros where they ar different
    ~~~


-   **Computing on data**

    ~~~octave
    A = magic(3) %sum of elements in each row or column are equal
    sum(A) %summarize each elements
    prod(A) %multiply each elements
    floor(A) %取整
    max(A, [], 1) %obtain max value of each row
    max(A, [], 2) %obtain max value of each column
    max(max(A)) %obtain the max value of all elements
    sum(A, 1) %obtain sum of column, get a row vector
    sum(A, 2) %obtain sum of row, get a colum vector
    flipud(eye(10)) %inner argument matrix should be a nxn matrix
    ~~~

-   **Plotting data**

    ~~~octave
    t = [0:0.01:0.98];
    y1 = sin(2*pi*4*t);
    plot(t, y1)
    hold on %displayed on a single graph
    plot(t, y1, 'r') %display in red line
    xlabel('time') %xlabel, ylabel, title, etc
    legend('sin', 'cos') %display a legend for the current axes using the specified strings as labels
    print -dpng 'myPlot.png' %save as picture
    close %close figure
    figure(1); plot(t, y1)

    subplot(2,10,1) %create a 2x10 plot, current 1st plot activate
    axis([0 1 1 2]) %change axis range, x from 0 to 1, y from 1 to 2
    clf %clear figures
    ~~~

-   **Control statement: for, while, if statements**

    ~~~octave
    v = zeros(10,1);
    for i = 1: 10,
    	v(i) = 2^i;
    end 

    i = 1;
    while i<=5,
    	v(i) = 100;
    	i = i + 1;
    end 

    if i >= 3,
    	xxx
    else,
    	xxx
    end

    fprint('now you have almost finished basic octave learning. \n');
    pause; %suspend the execution for N seconds, e.g. pause (10)
    ~~~

-   **Vectorial implementation**
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
          a = 1/m*sum(X(:,1)'(X*theta-y));
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

-   17.10.02 detail octave tutorial
-   17.09.10 init create @draachen

Reference

-   [Andrea Ng: Machine Learning](https://www.coursera.org/learn/machine-learning)
-   [Octave documentation pages](http://www.gnu.org/software/octave/doc/interpreter/)