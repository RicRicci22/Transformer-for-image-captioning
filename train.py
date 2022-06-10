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
from itertools import chain

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
    value_to_idx,idx_to_value,ignored_words = word_frequency_ucm_uav(list(chain(*train_sentences, *val_sentences)), 1, test_sentences)

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
        first_key = list(images.keys())[0]
        img_feature_size = images[first_key].shape[0]
    
    ################## HYPERPARAMETERS #####################
    learning_rate = 0.0005
    epochs = 10
    batch_size = 64
    max_len = 100 
    ######################################################## 

    trainset = Dataset_UCM(train_sentences,images,train_images_list,max_len=max_len)

    trainloader = DataLoader(trainset, batch_size=batch_size,shuffle=True,collate_fn=collate_fn)

    # Declare the model 
    model = Transformer(len(value_to_idx)+1,0,dropout=0.2) 
    # Send to cuda
    if(torch.cuda.is_available()):
        print('Model to CUDA!')
        model = model.cuda()
    # Define the loss and the optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0 
        for i, (images, targets, trg_masks) in enumerate(tqdm(trainloader)):
            # Feed the model
            y_pred = model(images,targets[:,:-1],trg_masks[:,:,:-1,:-1])
            y_pred = y_pred.reshape(-1,y_pred.shape[-1])
            # Loss calculation
            targets = targets[:,1:].reshape((-1,))
            loss = loss_fn(y_pred,targets)
            # Optimize
            optimizer.zero_grad()
            loss.backward()
            epoch_loss+=loss.item()
            optimizer.step()
        print("Epoch: %d,  loss: %.5f " % (epoch, epoch_loss/i))

    torch.save(model.state_dict(), r'transformer_captioning_resnet101.pt')
