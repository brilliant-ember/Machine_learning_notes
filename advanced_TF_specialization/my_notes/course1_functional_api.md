
### Sequential model example
![](screenshots/2021-10-05-04-21-03.png)
1. The model is return by the Sequential functin
2. The first layer is flattten to make the shape of the input into a single array so it can fit the the dense layer
3. the dense layer has 128 neurons, each has the relu activation functino
4. those 128 from previous layer are fed into 10 neurons of the next layer that has 10 neurons, each neuron is for an object that we classify. The softmax function means that we output a probability


### Rewriting with the functional API
![](screenshots/2021-10-05-04-28-31.png)



![](screenshots/2021-10-10-14-15-12.png)

![](screenshots/2021-10-10-14-56-59.png)


Multi input

![](screenshots/2021-10-11-12-40-11.png) ![](screenshots/2021-10-11-12-41-34.png)   ![](screenshots/2021-10-11-12-40-30.png)
![](screenshots/2021-10-11-12-41-46.png)


The functional API returns the output tensor if used in this format
`conv_output = Conv2D(filters=x, kernel_size=y, padding="same", activation="relu")(inputs)`

and will return the `keras.layers.Conv2D` object if used without the the final argument, so like this
`conv_output = Conv2D(filters=x, kernel_size=y, padding="same", activation="relu")`


The output tensor is just the tensor that results from math operations that involve the layer so would be in the case of a dense layer the multiplication of weights by the the activation, and adding bias

## Week two, custom loss functions

![](screenshots/2021-10-16-17-34-19.png)
![](screenshots/2021-10-16-17-44-55.png)
![](screenshots/2021-10-16-17-49-31.png)


If you want to add a hyperparameter to the loss functin, you gotta wrap it with another function.

-- Before
![](screenshots/2021-10-16-17-55-31.png)

-- After
![](screenshots/2021-10-16-17-55-55.png)


Now we can pass the threashold like this, and this enables you to update the threshold just like a hyper parameter.

![](screenshots/2021-10-16-17-55-00.png)


To turn the loss function into a class ![](screenshots/2021-10-16-18-01-20.png)
![](screenshots/2021-10-16-18-02-53.png)


## Week three custom layers

### Lamda layers

1. if you need to apply a simple function you can use a lambda layer, which provides a mapping from input to output ![](screenshots/2021-10-24-03-27-15.png)
- Look at this example below, we can replace the relu with a lambda
- ![](screenshots/2021-10-24-03-29-45.png) <br> Results are <br> ![](screenshots/2021-10-24-03-30-20.png)
- Now if we REMOVED the relu layer we see it makes the model perform worse ![](screenshots/2021-10-24-03-33-16.png)
- Now we add the Lamda layer and we see that the model is improved even more than with the relu ![](screenshots/2021-10-24-03-33-50.png) ![](screenshots/2021-10-24-03-34-04.png)
  
2. You can use any custom function in the lambda layer for example if we want to use a relu with a threshold ![](screenshots/2021-10-24-03-36-32.png)

### Custom layers

If you need a trainable layer or something advanced, then you can't use the lambda layer, you need a custom layer.

![](screenshots/2021-10-24-03-58-32.png)
![](screenshots/2021-10-24-03-59-10.png)
![](screenshots/2021-10-24-04-00-06.png)
![](screenshots/2021-10-24-04-07-40.png)
![](screenshots/2021-10-24-04-11-42.png)

Training 

![](screenshots/2021-10-24-04-13-22.png)
![](screenshots/2021-10-24-04-13-57.png)

After training

![](screenshots/2021-10-24-04-14-59.png)
![](screenshots/2021-10-24-04-15-13.png)


### Complex architectures with the Functional API

![](screenshots/2021-10-30-13-22-04.png)
![](screenshots/2021-10-30-13-22-22.png)
![](screenshots/2021-10-30-13-23-17.png)

To make the functional code clearner you can put it in a class
![](screenshots/2021-10-30-13-46-35.png)

Limitations of functional/sequential model, is that they are not dynamic and can't look back. More exotic models like recurssive models or dynamic ones can be hard to implement using func/seq models. So we use a subclass to make our implementation easier.

![](screenshots/2021-10-30-14-37-11.png)

Using custom model classes we can use them as blocks inside if statments easily, ResNet uses that approch, the residual blocks can skip layers depending on conditions.
![](screenshots/2021-10-30-15-14-30.png)

And instead of repeating ourselves we can use loops ![](screenshots/2021-10-30-15-16-09.png) ![](screenshots/2021-10-30-15-16-30.png)

![](screenshots/2021-10-30-15-29-57.png)

![](screenshots/2021-10-30-15-30-46.png)