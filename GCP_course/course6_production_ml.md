![](screenshots/2021-07-28-08-41-29.png)
reason why it is only 5% of the code base is in this next image ![](screenshots/2021-07-28-08-41-59.png)
![](screenshots/2021-07-28-08-42-37.png)
![](screenshots/2021-07-28-08-42-51.png)

TFRecord files are more fast than csv files.

If you have streaming data use PubSub, if using structured data from datawarehouse use bigQuery.

You should always validate the format of your data stream, if your model expects data from 0% to 100% and one system decides to give 0 to 1.0 you should detect that because it will throw off your model. You can check the data distribution. You should implement validation checks that answer any relevent questions about the data like the ones in teh image ![](screenshots/2021-07-28-08-50-51.png)

![](screenshots/2021-07-28-10-33-12.png)
IF your model has to be constant (like laws of physics, set in stone) static models are good. If your model has to be like fashion (changes constantly) then dynamic models are better. 

![](screenshots/2021-07-28-10-43-21.png)
![](screenshots/2021-07-28-10-43-55.png)
![](screenshots/2021-07-28-10-44-03.png)

there are 3 possible ways to do dynamic models on gcp
![](screenshots/2021-07-28-10-44-14.png)
![](screenshots/2021-07-28-10-44-53.png)

![](screenshots/2021-07-28-11-55-08.png)


![](screenshots/2021-07-28-12-10-40.png)


These are inside google cloud composer which is a managed version of ApacheAirflow
![](screenshots/2021-07-28-12-11-51.png)![](screenshots/2021-07-28-12-12-33.png)


![](screenshots/2021-07-28-12-57-32.png)





![](screenshots/2021-07-28-14-59-52.png)
you can't unlearn what the model has learned from poluted data. but you can roll back to previous backup of the model. That's why you should backup constantly.

Make sure your model doesn't memorize wrong things like people names, if you feed it a feature that includes ppl names and "actions" it might memorize that ppl with certain names tend to do certain actions. For example if you want to predict if a women is pregnant and in the training data it just so happens that a lot of pregnant women were named Fatima then the model will learn that which is wrong and it will not generlize well. Also make sure the model data is logical like if the women's age is 0 then she is no way preganant and that data should not be put in the training, it is probably a mis-inputted data or something and you should always put data validation checks to make sure that there're logical boundries since computers are dumb.


![](screenshots/2021-07-29-10-32-04.png)


Kubeflow is an opensource software whos goal it to ultimatly push you to GCP loool