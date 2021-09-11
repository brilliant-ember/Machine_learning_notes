import tensorflow
import numpy
from random import randint
import matplotlib.pyplot as plt

def check_gpu_tensorflow(cuda_only = True):
    '''checks to see if there's a gpu usign tensorflow api, and checks tensorflow, if cuda=True it will check if you have cuda gpu and not others'''
    is_gpu_available = len(tensorflow.config.list_physical_devices('GPU')) >= 1
    gpu_string = "CUDA"
    if (cuda_only == False):
        gpu_string = "GPU"
    print(f"is {gpu_string} available: {is_gpu_available}")
    print("Tensorflow version ", tensorflow.__version__)
    print("Keras version ", tensorflow.keras.__version__)
    
    
# generate a random sequence of integers
def generate_sequence(length, num_unique):
    '''
    ARGS : length of sequence, number of unique digits => [list]
    generates a random sequence of integers, num_unique is the number of unique digits in the sequence, for example if num_unique is 2 the sequence will only be '1' and '0' as those are
    two unique digits'''
    
    return [randint(0, num_unique - 1) for _ in range(length)]

def plot_training_validation_accuracy_loss(history):
    '''takes the history map returned from the model.fit function and
    plots the validation and training loss-accuracy plots '''
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt_epochs = range(1, len(acc)+1)

    plt.figure(facecolor='w')
    plt.plot(plt_epochs, acc, 'b', label='Training_accuracy')
    plt.plot(plt_epochs, val_acc, 'r', label="Validation_accuracy")
    plt.title("Training and validation accuracy")
    plt.legend()
    plt.grid()
    plt.figure(facecolor='w')
    plt.plot(plt_epochs, loss, 'b', label="Training loss")
    plt.plot(plt_epochs, val_loss, 'r', label="Validation loss")
    plt.title("Training and validation loss")
    plt.legend()
    plt.grid()