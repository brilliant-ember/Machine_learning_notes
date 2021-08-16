# Course 4 intro to tensorflow
https://www.coursera.org/learn/intro-tensorflow/lecture/t6nP6/intro-to-course


tensorflow is a numerical computation lib so can be used instead of matlab or numpy to do GPU computing, aha so useful. U can use it to solve partial diffintial equations
![](screenshots/2021-07-19-07-34-52.png)

![](screenshots/2021-07-19-07-35-10.png)

TF uses directed graphs to represent computations called Directed Acyclic Graph aka DAGs. TF uses DAGs because the DAG is languague independant of the code and can be portable between different devices, meaning you write your DAG in python and store it in a saved model restore in C++ for low latency which can run on both GPU and CPU.

This is very similar to how Java and JVM (java virtual machine ) work. You write your code in a high level langaague (Java) and then your java code can be executed anywhere by the JVM. The JVM is OS specific and can talk to the hardware,usually software devs only have to know Java and for the JVM it is already made. In tensorflow you write your code in a highlevel langauge (Python) and the code is executed by the tensorflow runtime engine.

In TensorFlow, tensors are multi-dimensional arrays with a uniform type (called a dtype).  All tensors are immutable like Python numbers and strings: you can never update the contents of a tensor, only create a new one.  

![](screenshots/2021-07-19-07-50-36.png)


![](screenshots/2021-07-19-12-08-09.png)

![](screenshots/2021-07-19-12-09-59.png)
![](screenshots/2021-07-19-12-14-08.png)

![](screenshots/2021-07-19-12-14-28.png)

![](screenshots/2021-07-19-12-15-34.png)

![](screenshots/2021-07-19-12-16-28.png)

![](screenshots/2021-07-19-12-20-41.png)


<br>
from the lab find training-data-analyst > courses > machine_learning > deepdive2 > introduction_to_tensorflow > labs, and open tensors-variables.ipynb. at `git clone https://github.com/GoogleCloudPlatform/training-data-analyst`

there are shortcuts for matrix multiplication and operation a @ b is matmul, a*b is dot product 
![](screenshots/2021-07-19-18-32-15.png)


![](screenshots/2021-07-19-18-37-14.png)


<br>

##  About Axis

rank 1 tensor has 1 axis and rank 2 tensor has 2 axes and so on, axis means "diminsion" or how many nested arrays we have BUT axes doesnt mean rank, if the shape is (2,3,3)then each of the "2,3,4" is an axis.
![](screenshots/2021-07-19-18-39-39.png)

![](screenshots/2021-07-19-18-29-12.png)

## About shapes

![](screenshots/2021-07-19-18-45-29.png) ![](screenshots/2021-07-19-18-46-38.png)![](screenshots/2021-07-19-18-48-10.png)![](screenshots/2021-07-19-18-49-17.png)![](screenshots/2021-07-19-18-50-40.png)


## Broadcasting

![](screenshots/2021-07-19-18-56-04.png)![](screenshots/2021-07-19-18-56-56.png) nicer view![](screenshots/2021-07-19-18-57-31.png)

## Ragged tensor
![](screenshots/2021-07-19-19-00-20.png)

## Variables

![](screenshots/2021-07-19-19-40-28.png)

![](screenshots/2021-07-19-19-41-34.png)![](screenshots/2021-07-19-19-42-45.png)![](screenshots/2021-07-19-19-43-00.png)


![](screenshots/2021-07-19-19-46-01.png)


![](screenshots/2021-07-21-20-30-50.png)

![](screenshots/2021-07-21-20-36-16.png)
![](screenshots/2021-07-21-20-36-28.png)

Onehot encoding is not scaleable to huge datasets, that's why we use embedded columns, where one cell can have the "one hot encoding" instead of having it be whole columns


<br>

which is super useful

`tf.feature_column` a feature column provides methods for the input data to be properly transformed before sending it to a model for training. Again the model just wants to work with numbers, that's the tensors part.

followed along the code and made this notebook tensorflow_feature_col.ipynb


TF.data api allows you to build complex data pipeline and preprocessing, tf.data makes it possible to use in and out of memory files and process data in parallet  and even cache data.
![](screenshots/2021-07-25-08-19-05.png)

There are two ways to make a dataset.
	1. A datasource makes the dataset from data in memory
	2. a dataset from multiple tf.dataset objects. 
   Large datasets tend to be broken down into multiple files.

   there are specccialized dataset types
   1. TextLineDataset to make a dataset that uses textfiles
   2. TFRecordDataset uses TFRecord files
   3. FixedlengthRecordDataset uses a dataset from fixedlength record or binary files
   4. anything else use the generic dataset class and add your own encoding decoding

![](screenshots/2021-07-25-08-40-57.png)

![](screenshots/2021-07-25-08-43-54.png)

you can use TFRecord and TFexample which are protobuffers, as well as the data api to create input piptlines

![](screenshots/2021-07-25-09-07-54.png)
![](screenshots/2021-07-26-07-02-42.png)

To explore the data tensorflow offers Facets and Tf data validation https://www.tensorflow.org/tfx/data_validation/get_started

A good resource for a lot of things AI.


---

Both of these models are linear, in fact the more complex one is EXACTLY the same linear model even though there's a hidden layer of neruons ![](screenshots/2021-07-26-07-51-56.png) ![](screenshots/2021-07-26-07-52-35.png)


Even this one is still the same and it is linear ![](screenshots/2021-07-26-07-55-13.png) 

No matter how many hidden layers we add the model will always stay linear, all the hidden layers can collapse into a single layer becuase it will be a linear combination, this is because whhat the hidden layers is add matrix multiplication and addition which are linear operations.

To make nonlinear models you add non-linear activation functions!!

Training with Relu is faster than most other activation functions, but Relu suffers from zero negative domain problem. Signmoid doesn't have that problem but it has the vanishing gradiant problem.

![](screenshots/2021-07-26-08-01-01.png)

There are many variations of relu that helps us combat the negative zero domain problem. (ie if the grad is negatibve it will be zero causeing the weight to be zero and there will be a lot of zeros in the layer making it fail to train as the weights will not update and will go to zero)

![](screenshots/2021-07-26-08-02-46.png)
![](screenshots/2021-07-26-08-02-58.png)
![](screenshots/2021-07-26-08-03-31.png)
![](screenshots/2021-07-26-08-04-53.png)

![](screenshots/2021-07-26-08-05-21.png)

![](screenshots/2021-07-27-06-42-01.png)

Batch normalization can help spped up trainng times and solve innternal coveriance shift( how the signal changes as data goes through the network)



![](screenshots/2021-07-27-06-55-44.png)
![](screenshots/2021-07-27-06-57-00.png)



### Keras

- A sequential model is a convieniant way to create a stack of layers, where you have one input tensor and one output tensor. It is not good for things with multiple inputs and outputs, or a layer can have multiple inputs/outputs, there is layer sharing, or the model has a non linear topology (ie the model isnot just a stack of layers such as residual connection or multibranch)


- The more layers the more "patterns" your model can learn, but watch out because the more layers can also increase the likelyhood of overfitting as you might memorize all the patterns of the train dataset.

- adam and ftrl are good goto optimmizers
- ![](screenshots/2021-07-27-07-08-08.png)
- ![](screenshots/2021-07-27-07-12-52.png)
- The .fit() function works well for small datasets which can fit entirely in memory. However, for large datasets (or if you need to manipulate the training data on the fly via data augmentation, etc) you will need to use .fit_generator() instead. The .train_on_batch() method is for more fine-grained control over training and accepts only a single batch of data.

![](screenshots/2021-07-27-14-56-53.png)
![](screenshots/2021-07-27-14-57-16.png)


![](screenshots/2021-07-27-15-09-01.png)
![](screenshots/2021-07-27-15-09-34.png)
![](screenshots/2021-07-27-15-10-06.png)
![](screenshots/2021-07-27-15-17-16.png)
rnn cant be done in functional api


![](screenshots/2021-07-27-15-36-00.png)

usually a dropoff probability of between 20 to 50% is good.

![](screenshots/2021-07-27-15-38-13.png)

The Wide part of the model is associated with the memory element. In this case, we train a linear model with a wide set of crossed features and learn the correlation of this related data with the assigned label. The Deep part of the model is associated with the generalization element where we use embedding vectors for features. The best embeddings are then learned through the training process. While both of these methods can work well alone, Wide & Deep models excel by combining these techniques together. checkout this research paper https://arxiv.org/abs/1606.07792