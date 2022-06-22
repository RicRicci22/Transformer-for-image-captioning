# Transformer-for-image-captioning

## Intro 
In this repo I'll try to build a transformer for image captioning. I will train it on the UCM dataset, a famous dataset of remotely sensed images.

The transformer architecture can be adapted for image captioning, by tweaking the encoder part and feed image information as a serie of tokens. In this repo, the first strategy will be based on a ResNet-152, one of the last feature maps produced by the network is then flattened and used as a serie of tokens to be fed in the encoder. 
The second strategy will employ a pre-trained visual transformer (ViT), cloned by this repo https://github.com/lukemelas/PyTorch-Pretrained-ViT. In this case the output is already a series of processed tokens. 
The image is divided into patched that are fed into the ViT, which proceeds to combine them using the transformer paradigm. The output is then an equal length serie of output tokens, which are used for the prediction of the next word.

It can be noticed that here the problem is much more difficult with respect to translation problems, since the vocabulary of input tokens is of theoretically infinite (not infinite since pixels have discrete values) dimension, while in traslation even the biggest dictionaries can contain up to 100k words at most. 

This, combined with the fact that the same concept can be expressed by different images, results in a big challenge for transformer based captioning systems. 

## Dataset 
The images I will be using is the famous UCM dataset. The dataset can be downloaded from here https://mega.nz/folder/wCpSzSoS#RXzIlrv--TDt3ENZdKN8JA. 


## Usage 
To train the model, select the hyperparameters in the file train.py, then run python train.py 

