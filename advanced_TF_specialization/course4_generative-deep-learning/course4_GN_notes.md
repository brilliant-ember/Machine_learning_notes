# generative-deep-learning
look at my pytorch file it has better/similar explanation of style transfer

## Style transfer
![](screenshots/2022-02-27-09-54-55.png)


Approaches to style transfer:
1. Supervised learning: this is when you feed the network images and desired stylized images, and the network learns to produce the stylized image. Needs a lot of pairs of images for inputs and labels -> (original, stylized images) . Needs a lot of training data.
2. Neural Style transfer: Don't need to train the network to produce styles, instead, we extract features using a network and apply those as styles. Similar to transfer learning. ![](screenshots/2022-02-27-09-59-40.png)
3. Fast Neural Style transfer, it's a lot faster than neural style transfer


In CNNs the first few layers extract low level features, such as lines and simple shapes. 
The deeper levels extract higher level features such as faces, wheels, paws ...etc
So the style transfer usually maintains the higher level features but changes the lower level features to match the style source image. 

For example imagine you're drawing a bird, you can either choose a pencil or a paintbrush. The style of the drawing will depend on the drawing instrument you chose, however in both drawings you will have the bird's high lvl features such as beak, eyes, feathers, wings ...etc but what's different is the lower level features or the "stroke of the paint" features. ![](screenshots/2022-02-27-16-41-06.png)


## loss
We have two losses the style loss and the content loss. if the content loss is 0 then we have the original image. So we try to increase the content loss (counterintuitively) but not increase it too much that we don't recognize the image, but enough that the style and image merge together nicely.

![](screenshots/2022-03-02-07-35-11.png)![](screenshots/2022-03-02-07-54-01.png)
![](screenshots/2022-03-02-07-41-05.png) where alfa and beta are weights for each loss.


## Optional concepts
we don't need to understand it to solve the labs and assignments, but I want to learn it so here it is.
### The gram matrix
![](screenshots/2022-03-02-08-00-34.png) so the gram matrix is effectively a permutation of cols a1 and a2

In code it looks like this. Note that at the end we can get G using einsum notation which is a very 'short' way of typing everything that was typed above it ![](screenshots/2022-03-02-08-03-45.png)

### Einstein notation (einsum)
![](screenshots/2022-03-02-08-06-05.png) the `-> ij` mean that we want the output to have i rows and j columns.

![](screenshots/2022-03-02-08-10-24.png)![](screenshots/2022-03-02-08-11-30.png)![](screenshots/2022-03-02-08-11-45.png)![](screenshots/2022-03-02-08-12-27.png)

![](screenshots/2022-03-02-08-15-16.png)![](screenshots/2022-03-02-08-16-10.png)

There are many ways to do style transfer, one of them is the supervised approach which we're going to use in lab 1. ![](screenshots/2022-03-07-06-38-40.png)