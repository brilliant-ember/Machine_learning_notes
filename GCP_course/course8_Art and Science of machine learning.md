# Art and Science of machine learning

- Batch size is the number of samples the gradient is caluclated on
- ![](screenshots/2021-08-06-06-49-56.png)


- ![](screenshots/2021-08-06-06-51-46.png)

- we always randomize data cuz we dont want the model to memorize or learn the order
- ![](screenshots/2021-08-06-06-56-12.png)
- u need to know how many steps per epoch you need to traverse your dataset once, note that this will change depending on batch_size
- Google Vizier does auto hyperparameter tunning for us

- ![](screenshots/2021-08-06-10-00-21.png)
- ![](screenshots/2021-08-06-10-05-12.png)
- ![](screenshots/2021-08-06-15-47-17.png)
- Regularization can help make our model more "lean" and reduce train time
- regularization is a way to deal with lose, should we punish huge weights? or should we amplify others? things like that affect how we update our weights in the next epoch
- 
- L1 shrinks the less important feature’s coefficient to zero thus, removing some feature altogether. So, this works well for feature selection in case we have a huge number of features
- L1 increase sparcity to our model
- ![](screenshots/2021-08-06-11-56-22.png)
- ![](screenshots/2021-08-06-15-44-06.png)
- the more the inputs tehe bigger the model and the more time it takes to train, embeddings allow us to reduce teh number of inputs, emeding is  a way of endoding the input before inputting it to the model
- One-hot encoding is a type of embeddings
- i can use embeddings for my quraan classifier
- ![](screenshots/2021-08-06-22-00-13.png)