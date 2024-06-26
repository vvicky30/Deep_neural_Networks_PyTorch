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
#### PyTorch supports multiple types of tensors, including:-
1. FloatTensor: 32-bit float
2. DoubleTensor: 64-bit float
3. HalfTensor: 16-bit float
4. IntTensor: 32-bit int
5. LongTensor: 64-bit int

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
* Iterative Optimization: We want to minimise error as much as possible. We keep updating the parameters iteratively by Gradient Descent.


## TORCHVISION.DATASETS
All datasets are subclasses of torch.utils.data.Dataset i.e, they have __getitem__ and __len__ methods implemented. Hence, they can all be passed to a torch.utils.data.DataLoader which can load multiple samples parallelly using torch.multiprocessing workers. For example:
```
imagenet_data = torchvision.datasets.ImageNet('path/to/imagenet_root/')
data_loader = torch.utils.data.DataLoader(imagenet_data,
                                          batch_size=4,
                                          shuffle=True,
                                          num_workers=args.nThreads)
```

#### The following datasets are available:

* MNIST
* Fashion-MNIST
* KMNIST
* EMNIST
* QMNIST
* FakeData
* COCO
* Captions
* Detection
* LSUN
* ImageFolder
* DatasetFolder
* ImageNet
* CIFAR
* STL10
* SVHN
* PhotoTour
* SBU
* Flickr
* VOC
* Cityscapes
* SBD
* USPS
* Kinetics-400
* HMDB51
* UCF101
* CelebA

All the datasets have almost similar API. They all have two common arguments: transform and target_transform to transform the input and target respectively.

## For making costom-module :-
### CLASStorch.nn.Module
Base class for all neural network modules.
Your models should also subclass this class.
Modules can also contain other Modules, allowing to nest them in a tree structure. You can assign the submodules as regular attributes:
```
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, 5)
        self.conv2 = nn.Conv2d(20, 20, 5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        return F.relu(self.conv2(x))
```
## AUTOMATIC DIFFERENTIATION PACKAGE - TORCH.AUTOGRAD
torch.autograd provides classes and functions implementing automatic differentiation of arbitrary scalar valued functions. It requires minimal changes to the existing code - you only need to declare Tensor s for which gradients should be computed with the requires_grad=True keyword.
```
torch.autograd.backward(tensors, grad_tensors=None, retain_graph=None, create_graph=False, grad_variables=None)
```
Computes the sum of gradients of given tensors w.r.t. graph leaves.

The graph is differentiated using the chain rule. If any of tensors are non-scalar (i.e. their data has more than one element) and require gradient, then the Jacobian-vector product would be computed, in this case the function additionally requires specifying grad_tensors. It should be a sequence of matching length, that contains the “vector” in the Jacobian-vector product, usually the gradient of the differentiated function w.r.t. corresponding tensors (None is an acceptable value for all tensors that don’t need gradient tensors).

This function accumulates gradients in the leaves - you might need to zero them before calling it.
* Parameters
#### tensors (sequence of Tensor) –
Tensors of which the derivative will be computed.

#### grad_tensors (sequence of (Tensor or None)) –
The “vector” in the Jacobian-vector product, usually gradients w.r.t. each element of corresponding tensors. None values can be specified for scalar Tensors or ones that don’t require grad. If a None value would be acceptable for all grad_tensors, then this argument is optional.

#### retain_graph (bool, optional) –
If False, the graph used to compute the grad will be freed. Note that in nearly all cases setting this option to True is not needed and often can be worked around in a much more efficient way. Defaults to the value of create_graph.

#### create_graph (bool, optional) –
If True, graph of the derivative will be constructed, allowing to compute higher order derivative products. Defaults to False.

```
torch.autograd.grad(outputs, inputs, grad_outputs=None, retain_graph=None, create_graph=False, only_inputs=True, allow_unused=False)
```
Computes and returns the sum of gradients of outputs w.r.t. the inputs.

grad_outputs should be a sequence of length matching output containing the “vector” in Jacobian-vector product, usually the pre-computed gradients w.r.t. each of the outputs. If an output doesn’t require_grad, then the gradient can be None).

If only_inputs is True, the function will only return a list of gradients w.r.t the specified inputs. If it’s False, then gradient w.r.t. all remaining leaves will still be computed, and will be accumulated into their .grad attribute.

* Parameters
#### outputs (sequence of Tensor) – 
outputs of the differentiated function.

#### inputs (sequence of Tensor) – 
Inputs w.r.t. which the gradient will be returned (and not accumulated into .grad).

#### grad_outputs (sequence of Tensor) –
The “vector” in the Jacobian-vector product. Usually gradients w.r.t. each output. None values can be specified for scalar Tensors or ones that don’t require grad. If a None value would be acceptable for all grad_tensors, then this argument is optional. Default: None.

#### retain_graph (bool, optional) –
If False, the graph used to compute the grad will be freed. Note that in nearly all cases setting this option to True is not needed and often can be worked around in a much more efficient way. Defaults to the value of create_graph.

#### create_graph (bool, optional) –
If True, graph of the derivative will be constructed, allowing to compute higher order derivative products. Default: False.

#### allow_unused (bool, optional) –
If False, specifying inputs that were not used when computing outputs (and therefore their grad is always zero) is an error. Defaults to False.
