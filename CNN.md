# CNN
- conv netrual net and pooling are super important in comuter vision, and CNN can replace RNN and do a better job too!
- Conv layer + Pool layer break the image into small parts, then the following dense layer(fully connected) make descions based on those small parts the conv+pool layer detected ![](screenshots/2020-05-23-14-32-56.png) *https://youtu.be/2-Ol7ZB0MmU?t=775*

- A filters detect a certain thing in its size, for example if i have a 2x2 filter that detects diagonal lines and a 3x3 image, then the filter well tell me if each 2x2 chunck of the 3x3 image have a diagonal line, and we keep sliding this filter until we convered the entrie image and know where in the image we have diagonal lines
 ![](screenshots/2020-05-23-14-38-24.png) ![](screenshots/2020-05-23-14-40-00.png)*top is '\' filter, and bottom is '/' filter https://youtu.be/2-Ol7ZB0MmU?t=910*
 
 - The output of the conv+pool is this in the image, and if we pass that dense layer will tell us that this is an X !
  ![](screenshots/2020-05-23-14-41-28.png) *note that the pooled img is 2x2 which is smaller than 3x3 original https://youtu.be/2-Ol7ZB0MmU?t=953*

  ### a good resource to vizualize cnn https://poloclub.github.io/cnn-explainer/#article-pooling

## Maxpooling

  is when u reducec the image size based on the biggest pixel value
  ![](screenshots/2020-05-31-08-31-10.png) *MaxPooling https://youtu.be/o_DJ-FO6dw0?t=47*

  ![](screenshots/2020-05-31-19-11-55.png) *stride of 2x2 means that we shfit 2px horizantally, when we reach the end the top of the pool-window will slide 2px down*



![](screenshots/2020-05-31-08-38-28.png)*https://classroom.udacity.com/courses/ud187/lessons/f00868fe-5974-48c4-bf36-41c0372bed64/concepts/4f7b0b45-3d43-4daa-8d6a-054cdd14abe9*

<br>

## Convolution operation with RGB images 
![](screenshots/2020-05-31-15-39-55.png)*1 3d filter(kernal) per convoluted output https://youtu.be/iDH3LCZwL5M?t=265*
![](screenshots/2020-05-31-15-43-43.png)
![](screenshots/2020-05-31-15-44-27.png)*it is 1 image, and we apply many 3d filters on it, for each filter we get one 2d convoluted output (filtered image). Now after all filters are done we get the num of convoluted is images equal to the number of filtes used. We take all of those covoluted outputs and put them into one matrix(tensor) with depth equal to num_of_filters https://youtu.be/iDH3LCZwL5M?t=269*

### Oh btw the values in the filters(kernal)are random at first and get "trained" to produce the smallest possible error function

## Fighting Overfitting!

- Using Image augmentation we apply random transformations on the training images so our model sees more variation during training, maybe random zoom, random tilts and so on
  ![](screenshots/2020-05-31-20-21-58.png)*Image augmentaion examples https://youtu.be/Qgd7maIVytI?t=115*

- ### *DropOuts* A great way to help with reduce overfitting. Is when we turn off neurons during training, so other neurons can contrinute more
  



## Extra refs
- a guide on cnn https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53
- ways to reduce overfitting https://hackernoon.com/memorizing-is-not-learning-6-tricks-to-prevent-overfitting-in-machine-learning-820b091dc42