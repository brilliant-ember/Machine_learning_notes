# Lesson 5 notes 
# CNN


- The optimizer updates your weights
- when using a transfer learning model, we want to keep the feature detection layers the same, but only change the last output layer
- Tips from the end of the lesson
- ![](screenshots/2021-05-30-18-47-42.png)
- backpropagatino quantify how bad weights are in making a mistake, like how wrong is the weight
- Optimizeatino functino gives us a better weight value (gradient decent)
- Batch size is the number of images that will be seen in one training iteration, a training iteration is when we make a mistake and learn from it by backpropagation. 
- num of workers is if you want to load data in parallel
- in pytorch you have to define any layer with trainable weights in the `__init__()` function
- ![](screenshots/2021-05-30-19-54-37.png)
- 1 epoch means that you only see each 'image' in the training data only once, if you have 30 epochs means that you see your training data 30 times. the more epochs the closer you can get to overfitting
- creating a train and validation set ![](screenshots/2021-06-06-12-54-37.png)
- ![](screenshots/2021-06-06-12-58-57.png)


-Regular dense layers:
![](screenshots/2021-06-06-14-18-04.png)
- Everything is connected to everyting in a dense connected layer
- However in a CNN only certian areas of the image go to each neuron
- __CNNs__
- ![](screenshots/2021-06-06-14-18-20.png)
- ![](screenshots/2021-06-06-14-18-40.png)
- We can add more "feature" detecteros
- ![](screenshots/2021-06-06-14-17-03.png)
- ![](screenshots/2021-06-06-14-56-32.png)
- ![](screenshots/2021-06-06-14-56-55.png)
- ![](screenshots/2021-06-06-15-01-36.png)
- Edges are areas where intensity change quicly, in this case the high pass filter acts as an edge detector
- Notes from this youtube tuto https://youtu.be/IKOHHItzukk?t=263
- __Hyperparameters__ are experimental parameters that choosing is a kind of art in a CNN or Linear layers these are: in CNNs *kernal_size* , *out_channels* and in Linear layers *out_features*
- ![](screenshots/2021-06-06-22-15-02.png)
- 6 output channels means you have six filters example horizantal edge filter, vertical filter ...etc and the weights for the filters will be randolmly assigned at first then automatically trained
- The interface between the CNN and the linear layer here abides this formuls (this is why we have 4*4) the 12 is the output of the last CNN but the 4x4 comes from the formula in the img
- ![](screenshots/2021-06-06-22-21-51.png)
- ![](screenshots/2021-06-06-22-23-12.png)
- Pooling layers "condense" or "distile" the previous layer window to the next layer, this does throw away some information, spatial information to be exact. The using CNN with pooling we can detect features, but not how they relate to each other in space. A solution ot that is **Capsule networks**
- ![](screenshots/2021-06-13-12-08-15.png)
- ![](screenshots/2021-06-13-12-08-57.png)
- ![](screenshots/2021-06-13-12-10-47.png)
- For maxpooling the stride controles the downsample rate, a stride of 4 means that the image signal will be downsampled by a factor of 4

- ![](screenshots/2021-06-13-17-06-52.png)
- ![](screenshots/2021-06-13-17-07-20.png)
- ![](screenshots/2021-06-15-06-57-57.png)
- ### The CNN formula
- ![](screenshots/2021-06-15-07-00-35.png)


- ![](screenshots/2021-06-15-07-09-30.png)
- ![](screenshots/2021-06-15-07-04-48.png)
- ![](screenshots/2021-06-15-07-05-17.png)
- ![](screenshots/2021-06-15-07-08-29.png)
- ![](screenshots/2021-06-15-07-08-49.png)


- ![](screenshots/2021-06-15-07-11-42.png)