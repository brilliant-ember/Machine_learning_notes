Custom and Distributed Training with TensorFlow

https://www.coursera.org/learn/custom-distributed-training-with-tensorflow/lecture/poLBG/a-conversation-with-andrew-ng-overview-of-course-2

<style>
img{
  max-width: 80%;
}
</style>

#  Week 1
##  What is a tensor

![](screenshots/2021-11-14-13-31-02.png)

- types of tensors
  - `tf.Variable` can change (mutable),ex, `tf.Variable('Hello', tf.string)`
  - `tf.constant` cannot change, it is immutable, ex `tf.constant([1,2,3,4,5])`
  - `tf.constant([4,6])` --> `tf.Tensor([4 6], shape=(2,) , dtype=int32)` int32 is a 32 bit integer.
- ![](screenshots/2021-11-14-14-47-17.png)
- ![](screenshots/2021-11-14-14-49-03.png)
- ![](screenshots/2021-11-14-14-52-20.png)
- ![](screenshots/2021-11-14-14-52-57.png)
- You can't reshape a tf.Variable using the `shape` keyword argument. However, you can reshape a tf.constant using it!
- ![](screenshots/2021-11-14-14-54-29.png)


## Broadcasting, overloading and Numpy compatibility

- There are two way to execute code in tensorflow
  1. Graph based, when we put all code in a session in one graph and execute it in one go after we wrote all the code
  2. Eager based, when we execute the code line by line
- Broadcasting is performing ops on different shape tensors 
- tensors and numpy are compatible, and np-arrays and tf-tensors are interchangeable
- you can convert a tensor to np-array using `tensor.numpy()`
- When you inherit from the class Layer you have a property that lets you access all tensor class variables, regardless what they are called inside the class ![](screenshots/2021-11-16-04-43-15.png)


## Gradient Tape



### Few words about differentiation and gradient

Derivative is the rate of change at a certain point, in other words it is the slope at a point or how steep the slope is at point (x,y).


Gradient in the context of reducing loss. taking the gradient tells us if the function is headed up or down, we want it to go down to minimize loss.

Gradient is a vector, it points in the direction of the steepest slope. If you have a function of f(x,y,z) or even more than 3 variables, and you take the partial derivative of the function with respect to each variable you will have a vector telling you in which direction (x,y,z) the slope is steepest, and this is called the gradient.

The gradient can be either positive or negative depending on the sign of the gradient. Positive gradient tells you the steepest slope upwards and negative gradient tells you the steepest slope downwards.

Differentiation is the derivative operation, computers can do it numerically by using difference equations, or they can do it symbolically.

![](screenshots/2021-11-17-08-02-12.png) 
[Link](https://www.quora.com/What-is-the-difference-between-a-gradient-and-a-derivative)


### Back to gradient tape in tensorflow

The core of machine learning is optimizer functions that try to match features to labels by tweaking parameters. These optimizer functions work on _Gradient Decent_ and each optimizer function has its rate of convergence towards the optimal value (which is usually minimum loss).

In Tensorflow optimizer functions or algorithms are implemented using Tf automatic differentiation API called _Gradient Tape_ . This API lets you compute and track the gradient of every differentiable tensorflow operation
![](screenshots/2021-11-17-07-44-59.png).

The gradient tape knows which variable we can use to differentiate when we use the `trainable=True` keyword argument.

This function is `y=2x-1` and see if the model can learn if x=2 and b=-1.
![](screenshots/2021-11-17-08-06-13.png)
![](screenshots/2021-11-17-08-08-23.png)


This is the training for loop
![](screenshots/2021-11-17-08-09-45.png)


This is where the learning happens, in the fit_data function
![](screenshots/2021-11-17-08-12-51.png)

Note that we can use the tape variable outside the tape block in the image above

We want the derivative of the loss with respect to the weight w, so we do this as seen in the image above.
`w_gradient = tape.gradient(reg_loss, w)`
The negative of this gradient will point in the direction of optimal value for w, forming a very basic optimizer.

The assign sub that updates the parameter w and b, 
The math for back propagation is `w = w - gradient(loss,w)*learning rate` 
the assign_sub does that equation exactly.

![](screenshots/2021-11-23-19-38-13.png)


-- Comparing gradient tape with manual derivation


Operations within a gradient tape scope are recorded if at least one of their variables is being watched.
![](screenshots/2021-11-23-19-43-24.png)


Now we do the same thing manually:

![](screenshots/2021-11-23-19-47-07.png)
![](screenshots/2021-11-23-19-46-19.png)
![](screenshots/2021-11-23-19-48-04.png)
![](screenshots/2021-11-23-19-48-59.png)
![](screenshots/2021-11-23-19-49-24.png)


So in summary Gradient tape helps us do derivatives in a neat way.



When we call the `tape.gradient()` the resources occupied by the gradient are immediately released, to make the persist we use `persistant=True`
![](screenshots/2021-11-23-19-51-24.png)

![](screenshots/2021-11-23-19-53-18.png)


Here is a short piece of the doc string 

    ``` Python

    class GradientTape(persistent=False, watch_accessed_variables=True)
    ```

    Record operations for automatic differentiation.

    Operations are recorded if they are executed within this context manager and at least one of their inputs is being "watched".

    Trainable variables (created by tf.Variable or tf.compat.v1.get_variable, where trainable=True is default in both cases) are automatically watched. Tensors can be manually watched by invoking the watch method on this context manager.

    For example, consider the function y = x \* x. The gradient at x = 3.0 can be computed as:

    ``` python
    x = tf.constant(3.0)
    with tf.GradientTape() as g:
      g.watch(x)
      y = x * x
    dy_dx = g.gradient(y, x) # Will compute to 6.0
    GradientTapes can be nested to compute higher-order derivatives

    ```


## Week 2 Custom Training Loops

![](screenshots/2021-11-25-04-24-42.png)
![](screenshots/2021-11-25-04-23-11.png)
![](screenshots/2021-11-25-04-33-29.png)
![](screenshots/2021-11-25-04-34-19.png)

### How does gradients and derivatives help us to minimize loss ie how an optimizer function works.

A mean square loss function, has the shape of a square function since it is squared
![](screenshots/2021-11-25-04-37-14.png)

Say that the ball is the loss we got when we calculated the the mean square loss using the current weights and biases ![](screenshots/2021-11-25-04-38-38.png)

The negative gradient shows us the direction we should go to inorder ot reduce loss