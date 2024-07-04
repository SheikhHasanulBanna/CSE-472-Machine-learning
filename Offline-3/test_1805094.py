import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix,f1_score,precision_score,recall_score,accuracy_score

import torchvision.datasets as ds
from torchvision import transforms
import torch
import pickle

torch.manual_seed(94)

#set seed for reproducibility
np.random.seed(94)

class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input,isTrain):
        pass

    def backward(self, output_gradient, learning_rate):
        pass

class ReLu(Layer):
    def __init__(self):
        self.input = None

    def forward(self, input,isTrain=True):
        self.input = input
        return np.maximum(0,self.input)

    def backward(self, output_gradient, learning_rate):
        return np.multiply(output_gradient,np.where(self.input>0,1,0))
    
class Softmax(Layer):
    def forward(self, input,is_Train=True):
        tmp = np.exp(input-np.max(input, axis=1, keepdims=True))
        self.output = tmp / np.sum(tmp, axis=1, keepdims=True)
        return self.output
    
    def backward(self, output_gradient, learning_rate):
        # batch_size = self.output.shape[0]
        # print('batch_size: ====',batch_size)
        # print('output_gradient shape: ====',output_gradient.shape)
        gradients = np.zeros_like(output_gradient)
        for index, (single_output_gradient, single_output) in enumerate(zip(output_gradient, self.output)):
            single_output = single_output.reshape(-1, 1)
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            gradients[index] = np.dot(jacobian_matrix, single_output_gradient)
    
        return gradients
    
class Dropout(Layer):
    def __init__(self, rate):
        self.rate = 1-rate
        self.mask = None

    def forward(self, input,isTrain=True):
        if not isTrain:
            return input
        self.mask = np.random.binomial(1, self.rate, size=input.shape) / self.rate
        return input * self.mask

    def backward(self, output_gradient, learning_rate):
        return output_gradient * self.mask
    

class AdamLayer(Layer):
    def __init__(self, input_size, output_size):

        self.weights = np.random.randn( input_size,output_size) * np.sqrt(2. / input_size)
        self.bias = np.random.randn(1,output_size)
        self.m_weights = np.zeros_like(self.weights)
        self.v_weights = np.zeros_like(self.weights)
        self.m_bias = np.zeros_like(self.bias)
        self.v_bias = np.zeros_like(self.bias)
        self.t = 0

    def forward(self, input,isTrain=True):
        self.input = input

        output = np.dot(self.input,self.weights) + self.bias

        return output

    def backward(self, output_gradient, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-7):
        weights_gradient = np.dot(self.input.T,output_gradient)
        bias_gradient = np.sum(output_gradient,axis=0,keepdims=True)

        self.t += 1  # Increment Time Step

        # Compute biased first moment estimate for weights
        self.m_weights = beta1 * self.m_weights + (1. - beta1) * weights_gradient
        # Compute biased second raw moment estimate for weights
        self.v_weights = beta2 * self.v_weights + (1. - beta2) * np.square(weights_gradient)

        # Compute bias-corrected first moment estimate for weights
        m_hat_weights = self.m_weights / (1. - np.power(beta1, self.t))

        v_hat_weights = self.v_weights / (1. - np.power(beta2, self.t))

        # Same for biases
        self.m_bias = beta1 * self.m_bias + (1. - beta1) * bias_gradient
        self.v_bias = beta2 * self.v_bias + (1. - beta2) * np.square(bias_gradient)

        m_hat_bias = self.m_bias / (1. - np.power(beta1, self.t))
        v_hat_bias = self.v_bias / (1. - np.power(beta2, self.t))

        # Update weights and biases
        temp=learning_rate * m_hat_weights / (np.sqrt(v_hat_weights) + epsilon)
        self.weights -= temp

        self.bias -= learning_rate * m_hat_bias / (np.sqrt(v_hat_bias) + epsilon)

        input_gradient = np.dot(output_gradient,self.weights.T)
        return input_gradient
    
def cross_entropy(y_true, y_pred):
    samples = len(y_pred)
    y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

    if len(y_true.shape) == 1:
        correct_confidences = y_pred_clipped[range(samples), y_true]
    elif len(y_true.shape) == 2:
        correct_confidences = np.sum(y_pred_clipped * y_true, axis=1)
    
    negative_log_likelihoods = -np.log(correct_confidences)
    return negative_log_likelihoods

def cross_entropy_prime(y_true, y_pred):
    samples = len(y_pred)
    labels= len(y_pred[0])
    y_pred= np.clip(y_pred, 1e-7, 1 - 1e-7)
    if len(y_true.shape) == 1:
        y_true = np.eye(labels)[y_true]

    output_gradient = -y_true / y_pred

    return output_gradient/samples

train_validation_dataset = ds.EMNIST(root='./data', split='letters',
                              train=True,
                              transform=transforms.ToTensor(),
                              download=True)


independent_test_dataset = ds.EMNIST(root='./data', split='letters',
                             train=False,
                             transform=transforms.ToTensor())


X_Test,Y_Test= zip(*independent_test_dataset)
X_Test=np.vstack(X_Test)
Y_Test=np.array(Y_Test)-1
X_Test=X_Test.reshape(X_Test.shape[0],-1)
print('X_Test shape: ====',X_Test.shape)

def performance_calculation(target,output):
    #get accuracy, precision, recall, f1 score
    accuracy=accuracy_score(target,output)
    f1=f1_score(target,output,average='macro')
    #get confusion matrix
    confusion=confusion_matrix(target,output)

    return accuracy,f1,confusion

def predict(network, input,isTrain=True):
    output = input
    count=0
    for layer in network:

        output = layer.forward(output,isTrain)

        count+=1
    return output


#extract network from pickle file
with open('model2_0.0005.pkl', 'rb') as f:
    network = pickle.load(f)

# print('network: ====',network)

test_output = predict(network, X_Test,isTrain=False)
test_output_vector = np.argmax(test_output, axis=1)
test_accuracy,test_f1,test_confusion=performance_calculation(Y_Test,test_output_vector)
print('test_accuracy: ====',test_accuracy)
loss = cross_entropy(Y_Test, test_output)
test_error = np.sum(loss) / len(loss)
print('test_error: ====',test_error)
print('test_f1: ====',test_f1)
