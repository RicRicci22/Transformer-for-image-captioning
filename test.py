# This script automatically test the model on the test set and print the results in a file called "prediction.txt"
from utils import *
import pickle
from model import Transformer   
import torch
from nltk.translate.bleu_score import corpus_bleu
from bert_score import score

def evaluate_model(descriptions, predicted_desciptions):
    # calculate BLEU score
    print('BLEU-1: %f' % corpus_bleu(descriptions, predicted_desciptions, weights=(1.0, 0, 0, 0)))
    print('BLEU-2: %f' % corpus_bleu(descriptions, predicted_desciptions, weights=(0.5, 0.5, 0, 0)))
    print('BLEU-3: %f' % corpus_bleu(descriptions, predicted_desciptions, weights=(0.33, 0.33, 0.33, 0)))
    print('BLEU-4: %f' % corpus_bleu(descriptions, predicted_desciptions, weights=(0.25, 0.25, 0.25, 0.25)))
    # Calculate Bert Score
    # Need to have list of str
    predicted_senteces_str = []
    for sentence in predicted_desciptions:
        predicted_senteces_str.append(' '.join(sentence))
    reference_sentences_str = []
    for group in descriptions:
        reference_sentences_str.append(' '.join(group[0]))
    P, R, F1 = score(predicted_senteces_str, reference_sentences_str, lang="en", verbose=True)
    print(f"BERT F1 score: {F1.mean():.3f}")
    

if __name__ == '__main__':

    # Load UCM dataset
    senteces_path = r'UAV_dataset\filenames\descriptions_UAV.txt'
    train_filenames_path = r'UAV_dataset\filenames\filenames_train.txt'
    val_filenames_path = r'UAV_dataset\filenames\filenames_val.txt'
    test_filenames_path = r'UAV_dataset\filenames\filenames_test.txt'

    sentences, max_len = convert_to_words_ucm(senteces_path)

    # Sentences is a dictionary with key as the image name and value as the sentence

    # Split into train_test_val 
    train_,val_,test_ = create_lists_ucm_uav(train_filenames_path,val_filenames_path,test_filenames_path)

    train_sentences = [sentences[i] for i in train_]
    test_sentences = [sentences[i] for i in test_]
    val_sentences = [sentences[i] for i in val_]

    # Create the dictionary with train and val sentences
    value_to_idx,idx_to_value,ignored_words = word_frequency_ucm_uav(list(chain(*train_sentences, *val_sentences)), 5, test_sentences)
    
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
    # Load image feaures matrix
    with open(r'UAV_dataset\images_uav.pkl','rb') as file:
        images = pickle.load(file)
        first_key = list(images.keys())[0]
        img_feature_size = images[first_key].shape[0]
    
    # Load the model 
    model = Transformer(len(value_to_idx)+1,0,max_length=30,backbone='vit')
    model.load_state_dict(torch.load(r'transformer_captioning_ViT_UAV_30epochs_0drop.pt'))
    
    if(torch.cuda.is_available()):
        print('Model to CUDA!')
        model = model.cuda()
        
    # Model in evaluation mode
    model.eval()
    
    print('Testing on the test set')
    
    with torch.no_grad():
        list_references = []
        list_generated_captions = []
        # Evaluating on the trainset 
        for image in train_:
            print('evaluating image '+str(image))
            reference = sentences[image]
            # Clean reference from startseq and endseq
            new_ref = []
            for sentence in reference:
                new_ref.append(sentence[1:-1])
            
            list_references.append(new_ref)
            
            generated_caption = torch.zeros((model.max_len),dtype=torch.long).cuda()
            generated_caption[0] = 1
            image_features = images[image].cuda()
            for i in range(model.max_len-1):
                trg_mask = torch.tril(torch.ones((model.max_len,model.max_len),dtype=torch.int8).expand(1,1,model.max_len,model.max_len)).cuda()
                pad_mask = (generated_caption!=0).unsqueeze(-2)
                trg_mask[0,0,:,:] = pad_mask & trg_mask[0,0,:,:]
                y_pred = model(image_features,generated_caption.unsqueeze(0),trg_mask)
                y_pred = torch.argmax(torch.softmax(y_pred.squeeze(0),dim=1),dim=1)
                generated_caption[i+1]=y_pred[i]
                #print(generated_caption[1:i+1])
                if(y_pred[i]==value_to_idx['endseq'] or i==model.max_len-2): 
                    list_generated_captions.append([idx_to_value[i] for i in generated_caption[1:i+1].tolist()])
                    break
                
        print('Trainset metrics')
        evaluate_model(list_references,list_generated_captions)
        # Evaluating on the test set
        list_references = []
        list_generated_captions = []
        for image in test_:
            print('evaluating image '+str(image))
            reference = sentences[image]
            # Clean reference from startseq and endseq
            new_ref = []
            for sentence in reference:
                new_ref.append(sentence[1:-1])
            
            list_references.append(new_ref)
            
            generated_caption = torch.zeros((model.max_len),dtype=torch.long).cuda()
            generated_caption[0] = 1
            image_features = images[image].cuda()
            for i in range(model.max_len-1):
                trg_mask = torch.tril(torch.ones((model.max_len,model.max_len),dtype=torch.int8).expand(1,1,model.max_len,model.max_len)).cuda()
                pad_mask = (generated_caption!=0).unsqueeze(-2)
                trg_mask[0,0,:,:] = pad_mask & trg_mask[0,0,:,:]
                y_pred = model(image_features,generated_caption.unsqueeze(0),trg_mask)
                y_pred = torch.argmax(torch.softmax(y_pred.squeeze(0),dim=1),dim=1)
                generated_caption[i+1]=y_pred[i]

                if(y_pred[i]==value_to_idx['endseq'] or i==model.max_len-2): 
                    list_generated_captions.append([idx_to_value[i] for i in generated_caption[1:i+1].tolist()])
                    break
        
        print('Testset metrics')
        evaluate_model(list_references,list_generated_captions)
    
    with open('list_predictions.pkl','wb') as file:
        pickle.dump(list_generated_captions,file)
        
    with open('prediction_UAV_ViT.txt', 'w') as file:
        for i in range(len(list_references)):
            file.write('References image '+str(test_[i])+'\n')
            for ref in list_references[i]:
                file.write(' '.join([i for i in ref])+'\n')
            file.write('Prediction\n')
            file.write(' '.join([i for i in list_generated_captions[i]])+'\n\n')