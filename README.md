# Deep_neural_Networks_PyToacrch
Building deep neural networks with the help of PyToarch
# Dynamic graph generation, tight Python language integration, and a relatively simple API makes PyTorch an excellent platform for research and experimentation.

## Installation
PyTorch provides a very clean interface to get the right combination of tools to be installed. Below a snapshot to choose and the corresponding command. Stable represents the most currently tested and supported version of PyTorch. This should be suitable for many users. Preview is available if you want the latest version, not fully tested and supported. You can choose from Anaconda (recommended) and Pip installation packages and supporting various CUDA versions as well.
![Screen shot](https://miro.medium.com/max/1400/1*fDlRnTbbi8_j82iw0VqxPw.png)

## PyTorch Modules
Now we will discuss key PyTorch Library modules like Tensors, Autograd, Optimizers and Neural Networks (NN ) which are essential to create and train neural networks.

### 1. Tensors
Tensors are the workhorse of PyTorch. We can think of tensors as multi-dimensional arrays. PyTorch has an extensive library of operations on them provided by the torch module. PyTorch Tensors are very close to the very popular NumPy arrays . In fact, PyTorch features seamless interoperability with NumPy. Compared with NumPy arrays, PyTorch tensors have added advantage that both tensors and related operations can run on the CPU or GPU. The second important thing that PyTorch provides allows tensors to keep track of the operations performed on them that helps to compute gradients or derivatives of an output with respect to any of its inputs.

### 2. Autograd
Autograd is automatic differentiation system. What does automatic differentiation do? Given a network, it calculates the gradients automatically. When computing the forwards pass, autograd simultaneously performs the requested computations and builds up a graph representing the function that computes the gradient.
How is this achieved?
PyTorch tensors can remember where they come from in terms of the operations and parent tensors that originated them, and they can provide the chain of derivatives of such operations with respect to their inputs automatically. This is achieved through requires_grad, if set to True.
```
t= torch.tensor([1.0, 0.0], requires_grad=True)
```
After calculating the gradient, the value of the derivative is automatically populated as a grad attribute of the tensor. For any composition of functions with any number of tensors with requires_grad= True; PyTorch would compute derivatives throughout the chain of functions and accumulate their values in the grad attribute of those tensors.

### 3. Optimizers
Optimizers are used to update weights and biases i.e. the internal parameters of a model to reduce the error. Please refer to my another article for more details.
PyTorch has an torch.optim package with various optimization algorithms like SGD (Stochastic Gradient Descent), Adam, RMSprop etc .
Let us see how we can create one of the provided optimizers SGD or Adam.
```
import torch.optim as optim
params = torch.tensor([1.0, 0.0], requires_grad=True)
learning_rate = 1e-3
## SGD
optimizer = optim.SGD([params], lr=learning_rate)
## Adam
optimizer = optim.Adam([params], lr=learning_rate)
```
Without using optimizers, we would need to manually update the model parameters by something like:
 ```
 for params in model.parameters(): 
       params -= params.grad * learning_rate
```
We can use the step() method from our optimizer to take a forward step, instead of manually updating each parameter.
```
optimizer.step()
```
The value of params is updated when step is called. The optimizer looks into params.grad and updates params by subtracting learning_rate times grad from it, exactly as we did in without using optimizer.
torch.optim module helps us to abstract away the specific optimization scheme with just passing a list of params. Since there are multiple optimization schemes to choose from, we just need to choose one for our problem and rest the underlying PyTorch library does the magic for us.

### 4. Neural Network
In PyTorch the torch.nn package defines a set of modules which are similar to the layers of a neural network. A module receives input tensors and computes output tensors. The torch.nn package also defines a set of useful loss functions that are commonly used when training neural networks.
Steps of building a neural network are:
* Neural Network Construction: Create the neural network layers. setting up parameters (weights, biases)
* Forward Propagation: Calculate the predicted output. Measure error.
* Back-propagation: After finding the error, we backward propagate our error gradient to update our weight parameters. We do this by taking the derivative of the error function with respect to the parameters of our NN.
* Iterative Optimization: We want to minimize error as much as possible. We keep updating the parameters iteratively by gradient descent.

