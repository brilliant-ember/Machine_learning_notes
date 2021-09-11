# Embedding

We want to represent text as integers so out ML can read it.

we can use one hot encoding, but then  the matrix will be HUGE depending on how many dimentions you have.
So we use an embedding layer. An embedding layer trains with your model it basically finds relationships in your data words that are related to each other will be encoded as vecotors that are closer to each other in the 3d space.

It is recommended to use Embedding layer whenever you use an RNN even if it is not encoding text, it is perfered over one hot encoding becasuse it acheives way better results, and the embedding layer can be saved an used in other models that are related. The embedding layer takes the number of unique tokens plus 1, always plus 1. and then it takes the output dimensions, which is somewhat arbitary. the output dimention is a hyper parameter that can have a big sway in the final accuarcy of the model.

`model.add(Embedding(num_unique_tokens+1, output_dimintion))`
