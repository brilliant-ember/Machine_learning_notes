# Encoder Decoder archetecture


[video link](https://youtu.be/iHJkfsV9cqY?list=PLQflnv_s49v-4aH-xFcTykTpcyWSY4Tww&t=1133)


![](screenshots/2021-09-17-06-55-14.png)

There are other models to solve seq2seq problems such as convolutional and reinforcement models, but for now we will stick with RNN.

For an LSTM-RNN here are the possible return cases
we return the state higlighted in red for each case

1. Default![](screenshots/2021-09-17-07-04-38.png) <br> <br> <br>
2. Return Sequences = true ![](screenshots/2021-09-17-07-07-04.png) <br> <br> <br>
3. return_states = true ![](screenshots/2021-09-17-07-08-33.png) <br> <br> <br>
4. return_sequnces=true and return_states=true ![](screenshots/2021-09-17-07-10-48.png)

![](screenshots/2021-09-17-07-14-48.png)
![](screenshots/2021-09-17-07-17-17.png)
![](screenshots/2021-09-17-07-19-17.png)
![](screenshots/2021-09-17-07-20-32.png)
![](screenshots/2021-09-17-07-29-12.png)
![](screenshots/2021-09-17-07-27-53.png)
![](screenshots/2021-09-17-07-31-05.png)
![](screenshots/2021-09-17-07-30-36.png)

![](screenshots/2021-09-17-07-32-55.png)
![](screenshots/2021-09-17-07-37-03.png)
![](screenshots/seq2seq-animation.gif)


We can have fixed or variable sequence length


Usual encoder-decoder model goes like this:
1. Encoder is just an LSTM block that receives the input data and outputs its cell states, these states are called context vector, or latent variables. The context vector is a representation of the input data and is the "bottleneck" of the input data, in a way it is an encoding of the input data.
2. The decoder which is just another LSTM runs on one sequences at a time, it takes the context vector as its inital state and the output of the last decoder run (for the intial run it takes the output of the lstm) and outputs its states and an output, the state and outputs are fed into the next decoder run until the end of the sequence.
3. There's a technique called teacher forcing where instead of the decoder just reciving the states and output data from previous decoder run, it also recieves the y-label data from the training dataset, this technique imporves training


