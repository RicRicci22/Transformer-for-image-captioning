import torch 
import torch.nn as nn 
from dataset import Dataset_UCM
import pickle
from utils import *

from dataset import collate_fn
from torch.utils.data import DataLoader
from model import Transformer   
import torch.optim as optim
from tqdm import tqdm
from custom_loss import custom_cross_entropy, weight_decreasing
from torchsummary import summary
import matplotlib.pyplot as plt 

if __name__ == '__main__':

    # Load UCM dataset
    senteces_path = r'UCM_dataset\filenames\descriptions_UCM.txt'
    train_filenames_path = r'UCM_dataset\filenames\filenames_train.txt'
    val_filenames_path = r'UCM_dataset\filenames\filenames_val.txt'
    test_filenames_path = r'UCM_dataset\filenames\filenames_test.txt'

    sentences, max_len = convert_to_words_ucm(senteces_path)

    # Sentences is a dictionary with key as the image name and value as the sentence

    # Split into train_test_val 
    train_,val_,test_ = create_lists_ucm_uav(train_filenames_path,val_filenames_path,test_filenames_path)

    train_sentences = [sentences[i] for i in train_]
    test_sentences = [sentences[i] for i in test_]
    val_sentences = [sentences[i] for i in val_]

    # Create the dictionary with train and val sentences
    value_to_idx,idx_to_value,ignored_words = word_frequency_ucm_uav(list(chain(*train_sentences, *val_sentences)), 5, test_sentences)
    train_sentences = []
    train_images_list = []
    for i in train_:
        for sentence in sentences[i]:
            processed_sentence = []
            for word in sentence:
                try:
                    processed_sentence.append(value_to_idx[word])
                except:
                    processed_sentence.append(value_to_idx['unk'])
                    
            train_sentences.append(processed_sentence)
            train_images_list.append(i)
            
    # Load image feaures matrix
    with open(r'UCM_dataset\images_ucm.pkl','rb') as file:
        images = pickle.load(file)
    
    ################## HYPERPARAMETERS #####################
    learning_rate = 0.0001
    lr_backbone = 0.00001
    epochs = 30
    batch_size = 64
    max_len = 30
    dropout = 0
    hidden_size = 256
    num_layers = 6
    forward_exp = 4
    heads = 8
    clip = 0.1
    init_w = 1
    backbone = 'resnet' # 'resnet' or 'vit' for visual transfomer
    ######################################################## 

    trainset = Dataset_UCM(train_sentences,images,train_images_list,max_len=max_len)
    trainloader = DataLoader(trainset, batch_size=batch_size,shuffle=True,collate_fn=collate_fn)

    # Declare the model 
    model = Transformer(len(value_to_idx)+1,0,dropout=dropout,hidden_size=hidden_size,num_layers=num_layers,forward_exp=forward_exp,heads=heads,max_length=max_len,backbone = backbone)
    # for name, param in model.named_parameters():
    #     print(name,param.shape)
    #     print(param.requires_grad)
    # Send to cuda
    if(torch.cuda.is_available()):
        print('Model to CUDA!')
        model = model.cuda()
        
    # Getting the list of different weight parts of the network
    backbone = []
    other = []
    for name,param in model.named_parameters():
        if('backbone' in name):
            backbone.append(param)
        else:
            other.append(param)
    # Define the loss and the optimizer
    # Create the weights so that to ignore the unk word
    weights = torch.ones((len(value_to_idx)+1,)) # Also account for the padding, but must be handled better! 
    weights[-1] = 0
    weights[0] = 0
    # Pass the weights to cuda if available
    if torch.cuda.is_available():
        weights = weights.cuda()
    loss_fn = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam([{'params':backbone}, {'params':other}] ,lr=learning_rate, betas=(0.9, 0.98), eps=1e-9)
    
    optimizer.param_groups[0]['lr'] = lr_backbone
    optimizer.param_groups[1]['lr'] = learning_rate

    model.train()
    for epoch in range(epochs): 
        epoch_loss = 0 
        for i, (images, targets, trg_masks) in enumerate(tqdm(trainloader)):
            # Feed the model
            y_pred = model(images,targets[:,:-1],trg_masks[:,:,:-1,:-1])
            y_pred = y_pred.reshape(-1,y_pred.shape[-1])
            # Create mask 
            targets_mask = torch.zeros((targets[:,1:].shape)).cuda()
            targets_mask[:,0] = 1
            # Loss calculation
            targets = targets[:,1:].reshape((-1,))
            targets_mask = targets_mask.reshape((-1,))
            loss = loss_fn(y_pred,targets)
            # weight = weight_decreasing(init_w,(init_w-1)/(epochs-5),epoch)
            # loss = custom_cross_entropy(y_pred,targets,targets_mask,weight)
            # Optimize
            optimizer.zero_grad()
            loss.backward()
            epoch_loss+=loss.item()
            # Clip the gradients 
            if(clip):
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
        print("Epoch: %d,  loss: %.5f " % (epoch, epoch_loss/i))

    torch.save(model.state_dict(), r'transformer_captioning_ViT_UCM_30epochs_0drop.pt')
