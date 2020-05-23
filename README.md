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
'''
t= torch.tensor([1.0, 0.0], requires_grad=True)
'''
After calculating the gradient, the value of the derivative is automatically populated as a grad attribute of the tensor. For any composition of functions with any number of tensors with requires_grad= True; PyTorch would compute derivatives throughout the chain of functions and accumulate their values in the grad attribute of those tensors.
