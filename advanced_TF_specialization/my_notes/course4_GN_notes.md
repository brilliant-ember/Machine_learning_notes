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

![](screenshots/2022-03-02-07-35-11.png)
![](screenshots/2022-03-02-07-41-05.png) where alfa and beta are weights for each loss.


## Optional concepts
we don't need to understand it to solve the labs and assignments, but I want to learn it so here it is.
### The Gram Matrix

For more in depth gram matrix stuff go to `pytorch/lesson6-notes-styleTransfer.md`

![](screenshots/2022-03-02-07-54-01.png)
![](screenshots/2022-03-02-08-00-34.png) so the gram matrix is effectively a permutation of cols a1 and a2

In code it looks like this. Note that at the end we can get G using einsum notation which is a very 'short' way of typing everything that was typed above it ![](screenshots/2022-03-02-08-03-45.png)

### Einstein notation (einsum)
![](screenshots/2022-03-02-08-06-05.png) the `-> ij` mean that we want the output to have i rows and j columns.

![](screenshots/2022-03-02-08-10-24.png)![](screenshots/2022-03-02-08-11-30.png)![](screenshots/2022-03-02-08-11-45.png)![](screenshots/2022-03-02-08-12-27.png)

![](screenshots/2022-03-02-08-15-16.png)![](screenshots/2022-03-02-08-16-10.png)

There are many ways to do style transfer, one of them is the supervised approach which we're going to use in lab 1. ![](screenshots/2022-03-07-06-38-40.png)


## Improving the style transfer

Now you may have noticed after lab1 that the stylized image has high frequency artifacts (ridges) that we'd like to remove. well, we can remove them with a high pass filter!
![](screenshots/2022-03-08-18-53-03.png) Observe the hard ridges in the stylized image.

![](screenshots/2022-03-08-18-56-43.png)



# Week 2 AutoEncoders
- They are neural networks capable of learning dense representations of input data  without supervision(training data is not labelled)
- They are good for dimensionality reduction and visualization
- can be used to as an efficient way to represent data.

For example take a look at these 2 sequences of numbers ![](screenshots/2022-03-24-18-27-21.png)

If you're asked to look at one of them and memorize it, which one is easier?

At first glance you might say the first one, but on a second though the second sequence is easier because there's a pattern, they're all squares, so you wouldn't have to remember them, you could just figure them out from the pattern.

The pattern is called _latent representation_ and the goal of auto-encoders is to create this latent representation or pattern.


 ![](screenshots/2022-03-26-08-18-54.png)
 ![](screenshots/2022-03-26-08-20-03.png)
and here's an example of how to do an autoencoder but with Sequential api
 ![](screenshots/2022-03-27-14-56-50.png)

 We can use an autoencoder that's built with CNN instead than one that's just built with Dense layers.

 Here is an autoencoder where the encoder and decoder are just built with dense layers. 
 
 The rows are in this order( original image, encodings, decodings)

 ![](screenshots/2022-03-27-14-53-14.png)


 And here's an autoencoder that uses CNNs which is a lot better than the dense only approach
 ![](screenshots/2022-03-27-14-53-41.png)

 ![](screenshots/2022-03-27-14-55-34.png)


## Denoising an image with autoencoder

We typically try to make the autoencoder model predict the input image itself.
![](screenshots/2022-03-27-16-07-05.png)

We can make the autoencoder `denoise` the image if we train it so it accepts a noisy image as input and the output is the original image without noise.
![](screenshots/2022-03-27-16-08-16.png)

![](screenshots/2022-03-27-16-08-39.png)

Not bad! it removes the noise!!

![](screenshots/2022-03-27-16-08-54.png)

We use `clip_by_value` to ensure that we don't accidentally pollute the normalized image, so we ensure the values are between 0 and 1

# Week 3 Variational AutoEncoder

The difference between an autoencoder and variational autoencoder (VAE) is that the latent space in the VAE is probabilistic. This allows us to create "new" data that look and feel randomized.


Regular autoencoder:
 ![](screenshots/2022-04-10-10-05-22.png)


 <br>

 Variational autoencoder (VAE):
 ![](screenshots/2022-04-10-10-06-52.png)


 * The notation for standard deviation is greek sigma: $\sigma$
* notation for mean is greek mu: $\mu$
* We will use gaussian probability density function aka normal distribution, denoted by letter N, so a normal distribution controlled by the mean and std-deviation is _N($\mu$, $\sigma$)_
* the mean and std-div will be learned by training, so we will express them as vectors that come from dense layers.

<br>

## The VAE encoder 
The encoder code structure will look like this ![](screenshots/2022-04-10-10-14-44.png)

and this is the first part of encoder implementation 
![](screenshots/2022-04-10-10-15-03.png)


The second part of the encoder will need us to use Gaussian distribution controlled by the mean and std-dev, one way to do this is to create a custom layer that will combine the mu and sigma with the gaussian noise, there're many ways to combine them with the noise, but a way that has proven effective is 
$$
   z = \mu + \epsilon * e^{0.5  \sigma}
$$
epsilon is the random sample.

as illustrated in the code below
![](screenshots/2022-04-10-10-25-43.png)

So the full encoder model will be 
![](screenshots/2022-04-10-10-26-44.png)

Note that in the encoder model we do return sigma and mu even though the decoder doesn't need them, this is because we will need them for the reconstruction loss later.

So now we have this part of the architecture
![](screenshots/2022-04-10-10-30-35.png)


next we will need the decoder, so let's see how to make it

<br>


## The VAE decoder

The decoder is this right part squared in red below

![](screenshots/2022-04-10-10-32-13.png)
![](screenshots/2022-04-10-10-32-42.png)
![](screenshots/2022-04-10-10-34-02.png)


<br>

## VAE Reconstruction loss

Now after decoding we need a way to make sure that our decoded data is good for reconstructing new data, we do this by creating reconstruction-loss

we usually use the Kullback-Leibler loss function in VAE

![](screenshots/2022-04-10-10-39-06.png)

![](screenshots/2022-04-10-10-39-40.png)


## Now to training the VAE
So we have our encoder, decoder, and loss
we're ready to start training!

here's the training loop (note this is valid for training the mnist dataset, could be different depending on your data)

![](screenshots/2022-04-10-10-44-26.png)
The bce lose is binary cross entropy 
a note about it from the course ![](screenshots/2022-04-10-10-48-42.png)


This is a sample of the data generated by our little VAE ![](screenshots/2022-04-10-10-50-31.png)