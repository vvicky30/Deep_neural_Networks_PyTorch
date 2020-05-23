# Deep_neural_Networks_PyToacrch
Building deep neural networks with the help of PyToarch
# Dynamic graph generation, tight Python language integration, and a relatively simple API makes PyTorch an excellent platform for research and experimentation.

## Installation
PyTorch provides a very clean interface to get the right combination of tools to be installed. Below a snapshot to choose and the corresponding command. Stable represents the most currently tested and supported version of PyTorch. This should be suitable for many users. Preview is available if you want the latest version, not fully tested and supported. You can choose from Anaconda (recommended) and Pip installation packages and supporting various CUDA versions as well.
![Screen shot](https://miro.medium.com/max/1400/1*fDlRnTbbi8_j82iw0VqxPw.png)

## PyTorch Modules
Now we will discuss key PyTorch Library modules like Tensors, Autograd, Optimizers and Neural Networks (NN ) which are essential to create and train neural networks.

# * Tensors
Tensors are the workhorse of PyTorch. We can think of tensors as multi-dimensional arrays. PyTorch has an extensive library of operations on them provided by the torch module. PyTorch Tensors are very close to the very popular NumPy arrays . In fact, PyTorch features seamless interoperability with NumPy. Compared with NumPy arrays, PyTorch tensors have added advantage that both tensors and related operations can run on the CPU or GPU. The second important thing that PyTorch provides allows tensors to keep track of the operations performed on them that helps to compute gradients or derivatives of an output with respect to any of its inputs.
