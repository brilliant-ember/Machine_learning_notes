# Advanced computer vision

<style>
img{
  max-width: 80%;
}
</style>

https://www.coursera.org/learn/advanced-computer-vision-with-tensorflow/lecture/e4mbd/what-is-transfer-learning

## Week 1 Transfer learning
it is using a pretrained model's weights to donwstream the trainig of our current model. For example if I alredy have a model that can classifiy cats and dogs, I can "downstream" that model's weights to train a new model that classifies cows and horses. Training from scratch is very demanding, transfer learning can help reduce time and can even improve accuracy

### Options in transfer learning
The closer the layer is to the input the more "general" it is, and the closer to the output the more it is "specific" features. So in general it is better to transfer the first CNN feature extractor layers and not the final dense classifier head
1. Freeze weights of cnn and use them as they are, and only train the classifier dense ![](screenshots/2021-12-04-19-56-38.png)
2. use the weights of the cnn as a starting point and train them even further along side the classifier which you're gonna train from scratch ![](screenshots/2021-12-04-20-01-13.png)


If we want to use transfer learning with ResNet  model and the CIFAR dataset

CIFAR images are 32x32 but the resnet model expects 224x224 so we need to use an upsampling layer
![](screenshots/2021-12-05-11-07-58.png)


### Evaluating object localization
if you want to detect where an object of intrest is in the image (localize it in the image) you need to define a loss value that defines how well you can predict where the object is in the image, ie you're trying to predict the bounding box around an object of interest, here's what the triaing data would look like ![](screenshots/2021-12-30-06-33-12.png)



The loss we use is IOU or intersection over union, which is a value from 0 to 1 telling us what's the percentage of intersection, if we predicted the bounding box (bbox) correctly we'll have a big intersection, if we did it wrong we will have no or little intersection

![](screenshots/2021-12-30-06-28-51.png)

![](screenshots/2021-12-30-06-31-06.png)



## Week 2 object detection and sliding windows

If you have an image that has multiple instances of the object you want to detect, and it also has empty space and other things that you don't want to detect you have to be smarter about how you do your object detection

![](screenshots/2021-12-27-08-04-28.png)

There are two stages to Object Detection:
1. Region proposal (what region of the image you will pass to your classifier, you can't pass all of it if it is a big image because that would be too inefficient)
2. Object detection/classification
   

There are few ways to detect an object within an image:
   1. Scanning the image with a sliding window that scans the image as the window slides, similar to convolution in how it scans. We try to classify what's in the window and ignore everything else :
      -  <span><img style="max-width:30%" src="screenshots/2021-12-27-08-37-42.png" /> <img style="max-width:30%" src="screenshots/2021-12-27-08-39-53.png"/> </span>
      - We can play with the size of the window
  
  2. Similar to sliding window but here we change the size of the image to work with less pixels, and then we can expand the image. Or we can calculate a union with different sized images detected objects to maximize accuracy
     - ![](screenshots/2021-12-27-08-45-49.png)
  3. Selective search, this method takes longer but it has good results. The algorithm makes proposals where the object is and then it combines the squares that contain the image
     - ![](screenshots/2021-12-27-08-47-43.png)

**R-CNN**
aka Regional CNN implements selective search on an image
![](screenshots/2021-12-27-08-51-48.png)


**RetinaNet**

Can predict bounding box around the image as well as classify the image. This model has its Model architecture separate from it's trained weights, this is often done in research. So we will have to load the model arch from a configuration file then load the weights from a checkpoint and combine it with the model if we want to use the model for transfer learning.

We want to retrain that pretrained model and load some of the pretrained weights saved in a checkpoint. All in order to predict and classify a novel class that wasn't part of the original training data of the model.

Since the model has 2 heads: 1. a bounding box predictor and a 2. classifier we will follow these steps to reuse the model:
1. take the base layers and the box prediction head as is
2. take the classification head and separate it from it's feature extractor, we will use the feature extractor only and discard the classification head 
3. then pass the model to checkpoint, then we will be able to restore pre-trained weights 
![](screenshots/2021-12-29-07-47-56.png)


We need to decide which trainable variables we're going to train, to see all trainable variables we can do code like this, note that RetinaNet is a huge model so we'll have a lot of trainable params, this is just a sample:
![](screenshots/2021-12-29-10-57-00.png)

To choose which ones to fine tone we do something like this. Remember that since the weights are already loaded form a checkpoint we don't have to train from scratch we will use the existing weight values as a starting point when we fine tone the trainable parameters we want.
![](screenshots/2021-12-29-10-58-37.png)



## Week 3 Image segmentation

![](screenshots/2022-01-02-10-34-33.png)

example  of training data that can classify the class "people"
![](screenshots/2022-01-02-10-35-08.png)


The encoder for image tasks is usually CNN, the earlier feature extraction layers extract low level  shapes such as lines,  but deeper layers in the CNN extract higher level features such as eyes .. etc
![](screenshots/2022-01-02-10-35-40.png)

![](screenshots/2022-01-02-10-42-30.png)

FCN = fully convolution network

This compares the different strides in the CNN feature extractor, the smaller the stride the more details we have.
![](screenshots/2022-01-02-10-44-30.png)

![](screenshots/2022-01-02-10-47-00.png)

This is an example of average pooling, we notice that the 2x2 pooling window with stride of 2x2 results in a downsampled pooled result  of half  the number of columns and rows of the input tensor


![](screenshots/2022-01-02-10-49-09.png)

![](screenshots/2022-01-02-10-49-22.png)


So if we pool 5 times  we reduce the image  by a factor of 2^5  ie  by 32 times, so  we need to upsample the image later by a factor of 32.


![](screenshots/2022-01-02-10-59-11.png)

![](screenshots/2022-01-02-11-00-21.png)

![](screenshots/2022-01-02-11-01-07.png)

![](screenshots/2022-01-02-11-01-54.png)


Transpose convolution (deconvolution) is a way to reverse convolution, but because of data loss in the pooling layers or lost in the edges the restored matrix will be close but not exactly the same  as the original matrix before convolution.

![](screenshots/2022-01-02-11-03-02.png)

Using 1D convolution with stride of becomes an easy way to reduce the dimensionality of our tensor since the only thing that changes is the number of filters. So 1D conv can act as a Flatten layer that can filter any arbitrary pixel size into 1x1xnum_filters, thereby it can be used to create networks that can accept input of any size! similar to how the FCN research paper made a network that can accept any image dimensions. 
![](screenshots/2022-01-02-13-41-25.png)

We can reduce the dimensionality from 64 to 32 as we see here (done in the camvid reduced dataset)

![](screenshots/2022-01-02-13-41-47.png)


### Evaluating IOU loss for segmentation
![](screenshots/2022-01-02-13-46-56.png)
![](screenshots/2022-01-02-13-47-09.png)
![](screenshots/2022-01-02-13-47-18.png)
![](screenshots/2022-01-02-13-48-18.png)
![](screenshots/2022-01-02-13-48-29.png)
![](screenshots/2022-01-02-13-48-55.png)

![](screenshots/2022-01-02-13-49-21.png)


![](screenshots/2022-01-09-11-00-52.png)


## Week 4 why does the model detect what it detected

Class activation gives us a 'heatmap' of the areas in the image that had the highest impact on the detection, in this image the darker pixels are the pixels that contributed most heavily on the prediction ![](screenshots/2022-01-14-18-05-55.png)

This is useful because it tells us what the model was looking at when it made the prediction, and this can help us determine why the model might have messed up, for example this cats vs dogs model learned to classify a cat vs dog based on the eyes of the animal, and when the eyes were not visible enough (one eye was hidden) the model predicted wrong, so we can use this insight to create a training data that forces the model to learn other features of the cat and dog, so may be have training images for the cats and dogs from behind.


How it works is as follows:
![](screenshots/2022-01-22-10-20-44.png)
![](screenshots/2022-01-22-10-36-39.png)

1. we first get the features from the last convolution layer in the model
2. we get the predicted category from the model (this is the output of the dense layer)
3. we grab the weights from the global average pooling layer, it is just before the classification dense layer  and it has one value per feature (learned weights for each feature), so we can get the weights corresponding to the prediction.
4. we scale the features (interpolate scale) to the required size
5. dot product the features and the weighting them from the class activation from the model to produce the cam (class activation map) 

So basically you take the outputs of the final convolution layer and then give them "importance' using weights
then zoom them up to the scale of the image, and finally overlay them on the image.

CAM can tell us what the model is looking at when it's making the prediction, for example this model below looks at the eyes to differentiate between a cat and a dog ![](screenshots/2022-01-24-02-19-26.png)

Now if for some reason the eyes are not visible the model will not accurately predict, for example  this image below there's only one eye, and the model doesn't even consider it, it looks at things on the left side of the image.
![](screenshots/2022-01-24-02-20-44.png)

The model looked at these features and not the eyes that it learned to use, so it predicted wrong, and even if it predicted right it would be a fluke.
![](screenshots/2022-01-24-02-21-11.png)

#### Saliency maps  

they are very similar to CAMs, but the way we create them is during the gradient step at comparing with the loss. Usually we take the gradient with respect to the model parameters (weights), but for saliency map we use take a grad
of the class score (loss) with respect to the input image, and the result is a an "activation map" that tells us how much a pixel contributed to the overall score.
![](screenshots/2022-01-24-02-34-14.png)![](screenshots/2022-01-24-02-34-25.png)


These saliency maps tell use that the model is looking at the subject that we want to detect. So this tells us that the model is looking at the right features.

![](screenshots/2022-01-24-02-36-05.png)![](screenshots/2022-01-24-02-36-13.png)


This map however seems that the model is looking at the pavement rather than the car we wanna detect. So our  model is not looking at the right thing and we need to change it.
![](screenshots/2022-01-24-02-38-38.png)

and here is a webapp that overlays the saliency map with the actual image ![](screenshots/2022-01-24-02-45-31.png)


how we take the saliency map:
![](screenshots/2022-01-24-02-47-06.png)

Now that we have the gradients there will be multiple filters in that last layer, so we collapse all that information into a single 2D matrix, we do this by reduce sum ny summing all the channels into one greyscale channel.

*There is an error in the image below* it should be de-normalized tensor, since we undo the normalization by multiplying by 255

![](screenshots/2022-01-24-03-21-49.png)

result with inception model (not same model as above)
 ![](screenshots/2022-01-24-03-23-58.png)


### Combining Saliency and CAMS

Cams looks at the final layer of convolution and figures which part of the image generated feature maps, while Saliency maps tells us which pixels in the input image were most impactful to the final classification (we did that by looking at the gradients of the final layers to see which ones had the steepest curve)

Now if we we combined the two we get GradCam model it's a CAM that uses the gradients of the final classifications. The GradCam model specifically tries to create a CAM and be more accurate where to place it by using saliency.
![](screenshots/2022-01-27-05-19-44.png)



### ZFNet
Visualization techniques pointed out weaknesses in AlexNet and helped researchers come up with ZFNet.

They used deconvolution for visualization. And had to unpool pooling operations, they did that by storing the location of the max-element in the kernel, they didn't save all the other values but at least they had the max-value and they knew where it was because they stored the location of the max-value in things they call _switches_
![](screenshots/2022-02-21-14-52-15.png)

The problems visualization with deconv exposed in AlexNet were mainly in layers 1 and layer 2. These problems were:
1. For layer 1 a mix of extremely high and low frequency was found, so there were no mid-frequencies. And this caused a chain-effect that the network only learned from low or high frequencies, making the learning poorer. ![](screenshots/2022-02-21-14-58-07.png)
2. Layer 2 Showed antialiasing which blures the outside of the features and are caused by the large strides used in the previous layer convolutions (antialiasing is caused when sampling freq is low) ![](screenshots/2022-02-21-14-59-46.png)


AlexNet got modified to reduce those weaknesses mentioned above:
1. Layer 1 filter size was reduced from 11x11 to 7x7
2. Stride was reduced to 2 instead of 4.

This is the result, we get more mid frequencies in layer 1, and cleaner features in layer 2
![](screenshots/2022-02-21-15-02-10.png)

