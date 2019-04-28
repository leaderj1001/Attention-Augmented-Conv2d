# Implementing Attention Augmented Convolutional Networks using Pytorch
- In the paper, it is implemented as Tensorflow. So I implemented it with Pytorch.

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
| Datasets | Model | Accuracy | Epoch | Training Time |
| :---: | :---: | :---: | :---: | :---: |
CIFAR-10 | WORK IN PROCESS | | |
CIFAR-100 | Just 3-Conv layers(channels: 64, 128, 192) | 61.6% | 100 | 22m
CIFAR-100 | Just 3-Attention-Augmented Conv layers(channels: 64, 128, 192) | 59.82% | 35 | 2h 23m

- I just want to see feasibility of this method(Attention-Augemnted Conv layer), I'll try about ResNet.
- The above results show that there are many time differences. I will think about this part a bit more.
  - I have seen the issue that the torch.einsum function is slow. [Link](https://github.com/pytorch/pytorch/issues/10661)
  - When I execute the example code in the link, the result was:<br><br>
  ![캡처](https://user-images.githubusercontent.com/22078438/56733452-2cc1c900-679b-11e9-861c-9aedfcedacac.PNG)
   - using cuda<br><Br>
   ![캡처](https://user-images.githubusercontent.com/22078438/56735393-4dd8e880-67a0-11e9-9fd0-6c0a4161d29d.PNG)
  
## Requirements
- tqdm==4.31.1
- torch==1.0.1
- torchvision==0.2.2


