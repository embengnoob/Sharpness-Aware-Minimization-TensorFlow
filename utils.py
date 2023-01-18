import resnet_cifar10
import tensorflow as tf
import matplotlib.pyplot as plt

# Reference 
# https://github.com/GoogleCloudPlatform/keras-idiomatic-programmer/blob/master/zoo/resnet/resnet_cifar10.py
def get_training_model(n_classes):
    # ResNet20
    n = 2
    depth =  n * 9 + 2
    n_blocks = ((depth - 2) // 9) - 1

    # The input tensor
    inputs = tf.keras.layers.Input(shape=(32, 32, 3))

    # The Stem Convolution Group
    x = resnet_cifar10.stem(inputs)

    # The learner
    x = resnet_cifar10.learner(x, n_blocks)

    # The Classifier for 100 classes
    outputs = resnet_cifar10.classifier(x, n_classes)

    # Instantiate the Model
    model = tf.keras.Model(inputs, outputs)
    
    
    return model

def plot_history(history, child_id):
    plt.figure(figsize=(18, 8)) 
    plt.plot(history.history["loss"], label="Training Loss")
    plt.plot(history.history["val_loss"], label="Validation Loss")
    plt.plot(history.history["accuracy"], label="Training Accuracy")
    plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
    plt.legend()
    plt.grid()
    plt.show()
    plt.savefig('history_plot_' + str(child_id) + '.png', bbox_inches='tight', dpi=350)