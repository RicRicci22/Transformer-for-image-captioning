# This script automatically test the model on the test set and print the results in a file called "prediction.txt"
from utils import *
import pickle
from model import Transformer   
import torch
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
    
    test_sentences = []
    test_images_list = []
    for i in test_:
        for sentence in sentences[i]:
            processed_sentence = []
            for word in sentence:
                try:
                    processed_sentence.append(value_to_idx[word])
                except:
                    processed_sentence.append(value_to_idx['unk'])
                    
                test_sentences.append(processed_sentence)
                test_images_list.append(i)
                
    # Load image matrix
    with open(r'UCM_dataset\images_ucm.pkl','rb') as file:
        images = pickle.load(file)
        first_key = list(images.keys())[0]
        img_feature_size = images[first_key].shape[0]
    
    # Load the model 
    model = Transformer(len(value_to_idx)+1,len(value_to_idx)+1,0,0)
    model.load_state_dict(torch.load(r'transformer_captioning_resnet152.pt'))
    
    if(torch.cuda.is_available()):
        print('Model to CUDA!')
        model = model.cuda()
        
    # Model in evaluation mode
    model.eval()
    
    print('Testing on the test set')
    
    list_references = []
    list_generated_captions = []
    
    with torch.no_grad():
        for image in test_:
            print('evaluating image '+str(image))
            reference = sentences[image]
            # Clean reference from startseq and endseq
            new_ref = []
            for sentence in reference:
                sentence.remove('startseq')
                sentence.remove('endseq')
                new_ref.append(sentence)
            
            list_references.append(new_ref)
            
            generated_caption = torch.zeros((1,model.max_len),dtype=torch.long).cuda()
            generated_caption[0,0] = 1
            image_features = torch.Tensor(images[image]).unsqueeze(0).cuda()
            for i in range(model.max_len-1):
                y_pred = model(image_features,generated_caption)
                print(y_pred.shape)
                y_pred = torch.argmax(torch.softmax(y_pred.squeeze(0),dim=1),dim=1)
                print(y_pred.shape)
                generated_caption[0,i+1]=y_pred[i]
                print(generated_caption)
                if(y_pred[i]==8):   # CHANGE WITH THE END TOKEN OF the VOCABULARY ! 
                    list_generated_captions.append([idx_to_value[i] for i in generated_caption[0,1:i+1].tolist()])
                    break
            print(list_generated_captions)
    
            
            
    
    print(list_references)
    print(list_generated_captions)