# Implementing Attention Augmented Convolutional Networks using Pytorch
- In the paper, it is implemented as Tensorflow. So I implemented it with Pytorch.

## Update (2019.05.11)
- Fixed an issue where key_rel_w and key_rel_h were not found as learning parameters when using relative=True mode.
- I have just modified attention-augmented-conv, and I will modify Wide-ResNet as soon as possible.<br><br>

- Example, relative=True, stride=1, shape=32
```python
import torch

from attention_augmented_conv import AugmentedConv

use_cuda = torch.cuda.is_available()
device = torch.deivce('cuda' if use_cuda else 'cpu')

tmp = torch.randn((16, 3, 32, 32)).to(device)
augmented_conv1 = AugmentedConv(in_channels=3, out_channels=20, kernel_size=3, dk=40, dv=4, Nh=4, relative=True, padding=1, stride=1, shape=32).to(device)
conv_out1 = augmented_conv1(tmp)
print(conv_out1.shape) # (16, 20, 32, 32)

for name, param in augmented_conv1.named_parameters():
    print('parameter name: ', name)
```
- As a result of parameter name, we can see "key_rel_w" and "key_rel_h".

- Example, relative=True, stride=2, shape=16
```python
import torch

from attention_augmented_conv import AugmentedConv

use_cuda = torch.cuda.is_available()
device = torch.deivce('cuda' if use_cuda else 'cpu')

tmp = torch.randn((16, 3, 32, 32)).to(device)
augmented_conv1 = AugmentedConv(in_channels=3, out_channels=20, kernel_size=3, dk=40, dv=4, Nh=4, relative=True, padding=1, stride=2, shape=16).to(device)
conv_out1 = augmented_conv1(tmp)
print(conv_out1.shape) # (16, 20, 16, 16)
```
- This is important, when using the "relative = True" mode, the stride * shape should be the same as the input shape. For example, if input is (16, 3, 32, 32) and stride = 2, the shape should be 16.

## Update (2019.05.02)
- I have added padding to the "AugmentedConv" part.
- You can use it as you would with nn.conv2d.
- I will attach the example below as well.
- Example, relative=False, padding=0
```python
import torch

from attention_augmented_conv import AugmentedConv

use_cuda = torch.cuda.is_available()
device = torch.deivce('cuda' if use_cuda else 'cpu')

temp_input = torch.randn((16, 3, 32, 32)).to(device)
augmented_conv = AugmentedConv(in_channels=3, out_channels=20, kernel_size=3, dk=40, dv=4, Nh=1, relative=False, padding=0).to(device)
conv_out = augmented_conv(tmp)
print(conv_out.shape) # (16, 20, 30, 30), (batch_size, out_channels, height, width)
```
- Example, relative=False, padding=1
```python
import torch

from attention_augmented_conv import AugmentedConv

use_cuda = torch.cuda.is_available()
device = torch.deivce('cuda' if use_cuda else 'cpu')

temp_input = torch.randn((16, 3, 32, 32)).to(device)
augmented_conv = AugmentedConv(in_channels=3, out_channels=20, kernel_size=3, dk=40, dv=4, Nh=1, relative=False, padding=1).to(device)
conv_out = augmented_conv(tmp)
print(conv_out.shape) # (16, 20, 32, 32), (batch_size, out_channels, height, width)
```
- Example, relative=True, stride=2, padding=1
```python
import torch

from attention_augmented_conv import AugmentedConv

use_cuda = torch.cuda.is_available()
device = torch.deivce('cuda' if use_cuda else 'cpu')

temp_input = torch.randn((16, 3, 32, 32)).to(device)
augmented_conv = AugmentedConv(in_channels=3, out_channels=20, kernel_size=3, dk=40, dv=4, Nh=1, relative=False, padding=1, stride=2).to(device)
conv_out = augmented_conv(tmp)
print(conv_out.shape) # (16, 20, 16, 16), (batch_size, out_channels, height, width)
```

- I added an assert for parameters (dk, dv, Nh).
```python
assert self.Nh != 0, "integer division or modulo by zero, Nh >= 1"
assert self.dk % self.Nh == 0, "dk should be divided by Nh. (example: out_channels: 20, dk: 40, Nh: 4)"
assert self.dv % self.Nh == 0, "dv should be divided by Nh. (example: out_channels: 20, dv: 4, Nh: 4)"
assert stride in [1, 2], str(stride) + " Up to 2 strides are allowed."
```


## I posted two versions of the "Attention-Augmented Conv"
  - Paper version is [here](https://github.com/leaderj1001/Attention-Augmented-Conv2d/blob/master/attention_augmented_conv.py)
  - AA-Wide-ResNet version is [here](https://github.com/leaderj1001/Attention-Augmented-Conv2d/blob/master/AA-Wide-ResNet/attention_augmented_conv.py)

# Reference
## Paper
- [Attention Augmented Convolutional Networks Paper](https://arxiv.org/abs/1904.09925)
- Author, Irwan Bello, Barret Zoph, Ashish Vaswani, Jonathon Shlens
- Quoc V.Le Google Brain
## Wide-ResNet
- [Github URL](https://github.com/meliketoy/wide-resnet.pytorch/blob/master/main.py)
- Thank you :)

## Method
![image](https://user-images.githubusercontent.com/22078438/56668731-ffb5dd80-66ea-11e9-9274-1223f579f039.PNG)

### Input Parameters
- In the paper, ![CodeCogsEqn (2)](https://user-images.githubusercontent.com/22078438/56719194-39cec000-677b-11e9-9ad9-4c58a65f41cd.gif)
 and ![CodeCogsEqn (3)](https://user-images.githubusercontent.com/22078438/56719216-44895500-677b-11e9-85ad-1c68dcae8435.gif)
 are obtained using the following equations.<br><br>
![CodeCogsEqn](https://user-images.githubusercontent.com/22078438/56719018-e3fa1800-677a-11e9-9393-1835b60c6fd0.gif), ![CodeCogsEqn (1)](https://user-images.githubusercontent.com/22078438/56719117-0b50e500-677b-11e9-84c8-73530191acb9.gif)

- Experiments of parameters in paper<br><br>
![캡처](https://user-images.githubusercontent.com/22078438/56719332-78fd1100-677b-11e9-9a26-b281fb2db7de.PNG)


## Experiments
- In the paper, they said that We augment the Wide-ResNet-28-10 by augmenting the first convolution of all residual blocks with relative attention using Nh=8 heads and κ=2, υ=0.2 and a minimum of 20 dimensions per head for the keys.

| Datasets | Model | Accuracy | Epoch | Training Time |
| :---: | :---: | :---: | :---: | :---: |
CIFAR-10 | Wide-ResNet 28x10(WORK IN PROCESS) | | |
CIFAR-100 | Wide-ResNet 28x10(WORK IN PROCESS) | | |
CIFAR-100 | Just 3-Conv layers(channels: 64, 128, 192) | 61.6% | 100 | 22m
CIFAR-100 | Just 3-Attention-Augmented Conv layers(channels: 64, 128, 192) | 59.82% | 35 | 2h 23m

- I don't have enough GPUs. So, I have many difficulties in training. Sorry... T.T
- I just want to see feasibility of this method(Attention-Augemnted Conv layer), I'll try about ResNet.
- The above results show that there are many time differences. I will think about this part a bit more.
  - I have seen the issue that the torch.einsum function is slow. [Link](https://github.com/pytorch/pytorch/issues/10661)
  - When I execute the example code in the link, the result was:<br><br>
  ![캡처](https://user-images.githubusercontent.com/22078438/56733452-2cc1c900-679b-11e9-861c-9aedfcedacac.PNG)
   - using cuda<br><Br>
   ![캡처](https://user-images.githubusercontent.com/22078438/56735393-4dd8e880-67a0-11e9-9fd0-6c0a4161d29d.PNG)
 
## Time complexity
- I compared the time complexity of "relative = True" and "relative = False".
- I'll compare the performance of the two different values(relative=True, relative=False).
- In addition, I will consider ways to reduce time complexity in "relative = True".<br>
![time_complexity](https://user-images.githubusercontent.com/22078438/57056552-376de800-6cde-11e9-90fc-492c28d78907.PNG)
  
## Requirements
- tqdm==4.31.1
- torch==1.0.1
- torchvision==0.2.2


