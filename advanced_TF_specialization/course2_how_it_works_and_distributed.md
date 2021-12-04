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
- ![](screenshots/2021-11-14-14-57-22.png)

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

The negative gradient shows us the direction we should go to in order to reduce loss
![](screenshots/2021-11-25-04-42-47.png)
![](screenshots/2021-11-25-04-48-26.png)
![](screenshots/2021-11-25-04-49-57.png)



<br>


- We use `SparseCategoricalCrossentropy()` when the category labels are integers and not one-hot encoded, it is a bit more efficent than regular `CategoricalCrossentropy()`

- Note: during batching if data doesn't batch evenly then the final batch will be "incomplete" and this will make the batch use data from other batches, this will create a bias or a skew in the loss calculation of the final batch

<br>

![](screenshots/2021-11-28-16-57-02.png)
![](screenshots/2021-11-28-17-01-42.png)
![](screenshots/2021-11-28-17-27-33.png)

![](screenshots/2021-11-29-01-27-07.png)

---

## Week 3 Graph model

![](screenshots/2021-12-02-03-24-10.png)
![](screenshots/2021-12-02-03-26-18.png)

Autograph takes python code and turns it into a graph

To generate graph code, write your function and then wrap it by using a special function from tensorflow,
you can do that using the decorator `@tf.function`, wrapping with a decorator allows us to "combine" two functions

For example 

``` python

@tf.function
def add(a,b):
  return a+b

```

Now the decorated add function has graph code, we can look at it by using
`print(tf.autograph.to_code(add.python_function))`


![](screenshots/2021-12-04-05-50-20.png)

Furthermore, any function called from within an annotated function will get called by using graph mode
![](screenshots/2021-12-04-05-51-55.png)


GraphMode speeds up code that has a lot of small ops for example this fizzbuzz game can be faster if it is using graph mode, as it has all those conditions and if-else flows.
![](screenshots/2021-12-04-05-55-35.png)


If we take a look at fizzbuzz graph code, it will be very hard to code
![](screenshots/2021-12-04-05-57-03.png)

Conditionals 

![](screenshots/2021-12-04-12-42-15.png)
![](screenshots/2021-12-04-12-48-56.png)

Regular print functions are not graph aware, but tf.print is
![](screenshots/2021-12-04-12-52-15.png)

Note that in graph based functions we need to sepreate the declaration of the variables and the function logic, othereise we run into errors:

<p><img src='screenshots/2021-12-04-12-54-32.png'/> <img src='screenshots/2021-12-04-12-55-15.png'/></p>
![](screenshots/2021-12-04-13-04-56.png)

however it is okay to use aliases like this
![](screenshots/2021-12-04-13-05-39.png)


## Week 4 Distribution stratigies

distributed training in tf is built around data parallelism, we replicate the model on different devices and each device has a different slice of the trainig data

![](screenshots/2021-12-04-13-13-20.png)

Some terms in distributed learning:
1. Device: cpu, gpu, tpu. 
2. Replica: a replica of our model, each replica runs in a sepreate device. for example, replicate the model on the cpu and gpu to train them in parallel
3. Worker: a piece of software running on the machine that trains the models on that machine (across multiple devices if applicable)
4. Mirrored variable: variables you want to keep sync across all replicas

### There are many distribuion stratigies
![](screenshots/2021-12-04-13-21-17.png)

In synchronous training all workers train at the same time and aggregate weights at each steps using an all-reduce algorithm

Asynchronous trainig all workers train independantly on input data and update variables using a parameter server.


![](screenshots/2021-12-04-13-25-57.png)


### Mirrored strategy
It is the most common strategy since devs are more likely to have a machine with multiple GPUs instead of a
network of devices
![](screenshots/2021-12-04-13-27-23.png)
![](screenshots/2021-12-04-13-29-58.png)


### Other strategies
1. single device `tf.distribute.OneDeviceStrategy` this uses a single machine but treats data as if it was distributed, this strategy is good if you want to test your distribution on a single machine first before you go on and try to distribute across others.
2. Multiworker mirrored strategy ![](screenshots/2021-12-04-15-00-41.png)![](screenshots/2021-12-04-15-01-53.png)
3. CentralStorageStrategy and Param server strategy ![](screenshots/2021-12-04-15-02-45.png)