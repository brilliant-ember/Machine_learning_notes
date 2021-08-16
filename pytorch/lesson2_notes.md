# udacity pytorch course notes

https://classroom.udacity.com/courses/ud188/lessons/b4ca7aaa-b346-43b1-ae7d-20d27b2eab65/concepts/6ba9c9eb-2e36-4b03-9bcc-01e71260a024

## Lesson 2
![](screenshots/2021-05-10-18-38-31.png)

![](screenshots/2021-05-10-18-39-27.png)

![](screenshots/2021-05-10-18-40-17.png)


The preseptron has the graph functino inside it, and the weights are just the linear function coefficients, the preceptron outputs a boolean 

![](screenshots/2021-05-10-18-37-10.png)

![](screenshots/2021-05-10-18-42-04.png)

![](screenshots/2021-05-10-18-42-57.png)

_Gradient descent_ is like being on top of a mount Errorist, and the hight you are in is the amount of error, you try to find a path that will reduce the hight, so you take a step in the direction the lowers the hight and you keep doing that until you're at the lowest possible place. 
GD works best with continous error, so you gotta define the error functino to output a continours value (like distance in a line), instead of a discrete one (like true or false).

![](screenshots/2021-05-13-02-43-20.png)

Changing to continous can be done by changing the activatino functino from step func to sigmoid
![](screenshots/2021-05-13-02-44-30.png)
![](screenshots/2021-05-13-02-46-15.png)
![](screenshots/2021-05-13-02-46-26.png)

btw the step activatino functino can be used if we want to make logic gates (AND, OR, XOR ...etc)


![](screenshots/2021-05-13-05-42-22.png) 

Notice how the probability always adds up to 1
![](screenshots/2021-05-13-05-44-15.png)


![](screenshots/2021-05-14-21-42-03.png)

cross entropy is to use log to turn multiplications into additinos, which are easier on CPU, the lowerst entropy means that the higher the propbability, so the less the entropy the better the model is.

Cross entropy mathmatically can tell us how similar to vectors are. If CE is low then the vecotrs are similar, if CE is high then the vectors are very different ![](screenshots/2021-05-15-04-49-59.png)

Exponent function is always positive, and Log functino can turn multiplications to additions thus making our work more efficient since multiplications are more expensive

* When traininig the model, the first "model" or weights are random, ML then uses gradient descent iteration to minimize the error **It is possible to get different training outcome each time you train!!** this is becaue the random initial placement of the weights plays some role.
  * Here is the same exact experiment, after each training we get a different result ![](screenshots/2021-05-15-11-47-23.png) ![](screenshots/2021-05-15-11-47-56.png)  ![](screenshots/2021-05-15-11-48-30.png)


* The difference btwn Gradient descent and preceptron algs is that GD gives us a number between 0 and 1 (analog) while the preceptron gives us a discrete 0 or 1 
    ![](screenshots/2021-05-15-11-50-07.png) ![](screenshots/2021-05-15-11-53-07.png) .
     That line is the model ![](screenshots/2021-05-15-11-54-27.png)
     Takes a point (x1, x2) and returns a probability of the line being correctly classified (how far into the blue region) ![](screenshots/2021-05-15-11-55-35.png)


## Nerual Network Architecture
The power of nerual networks (or multilayer preceptrons) come from their ability to deal with none linear data, so far our model was a stright line but what if it was a more complex shape?

* A complex thing is just putting more of simple things together, so to solve it we use many simple things together! We will use our simple linear model from before, only this time we will use more than one, then add them, and finally to get a probability between 0 and 1 we will use a sigmoid.![](screenshots/2021-05-15-12-01-53.png)
* We can control the output model by giving different weights to the indivitual models ![](screenshots/2021-05-15-12-03-21.png)
* The indivisual models are just a linear combination of the inputs times the weights plus a bias, and the output is just a linear combination of the previous models times the weights plus a bias
 ![](screenshots/2021-05-15-12-07-58.png)



 * using preceptrons, the 7 and 5 are the weights we assigned to each model ![](screenshots/2021-05-15-12-09-49.png) 
 *  breaking it down
 *  ![](screenshots/2021-05-15-12-12-12.png)
  ![](screenshots/2021-05-15-12-10-25.png)
* Cleaning up and joining them
    ![](screenshots/2021-05-15-12-12-44.png)
    ![](screenshots/2021-05-15-12-13-27.png)

    Using a different notation to present it ![](screenshots/2021-05-15-12-15-08.png)


<br>
 General architecture is 1. input layer, 2. many linear hidden layers (can also be nonlinear hidden layers), 3. an output layer which is just the combination of previous linear layers to create a final non linear layer

![](screenshots/2021-05-15-12-29-01.png)
![](screenshots/2021-05-15-12-29-54.png)
![](screenshots/2021-05-15-12-30-12.png)

* If we have N inputs then we have N diminetion models


* This is for a multilclassification problem
  ![](screenshots/2021-05-15-12-31-20.png)

* We can use nonlinear models as inputs to create an output nonlinear model! ![](screenshots/2021-05-15-12-32-33.png) 
  

Error function for one perceptron
![](screenshots/2021-05-16-08-40-25.png)

For multilayer perceptorn error is still the same since it is just a measure of how wrong a point is classified, but the predictino or y_hat is bit more complicated since it is a linear combi of sigmoids and weights npw ![](screenshots/2021-05-16-08-43-58.png)


Feedforward is the pass from inputs to output probability prediction, we use that prediction to get an error. Now if we want to correct for that error we use Backpropacation.
![](screenshots/2021-05-16-08-46-35.png)

---

### Math that you can ignore


![](screenshots/2021-05-16-08-56-05.png)

![](screenshots/2021-05-16-08-56-32.png)


![](screenshots/2021-05-16-08-59-42.png)

![](screenshots/2021-05-16-09-00-34.png)

![](screenshots/2021-05-16-09-03-15.png)

![](screenshots/2021-05-16-09-04-21.png)

---
---
---

Simple models tend to be better since they less prone to overfitting compared to complicated models

![](screenshots/2021-05-16-14-14-06.png)


![](screenshots/2021-05-16-14-16-31.png)

To fight overfitting we punish large weights useing one of two regularization functinos in the error function, by adding values that become bigger as weights get bigger, we have two reg functions L1 using sum of abs, and L2 using sum of squares
![](screenshots/2021-05-16-14-34-14.png)
![](screenshots/2021-05-16-14-34-34.png)

* To combat local minima in gradient decent one way is to start at different poinsts in the gradient and see which one is the global lowest![](screenshots/2021-05-16-14-38-55.png)


