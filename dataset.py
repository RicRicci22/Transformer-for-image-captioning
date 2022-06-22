import torch

class Dataset_UCM:
    def __init__(self, sentences, images, images_idx, max_len):
        # The targets are automatically inferred from the input data
        # The idea is that we shift one step to the right, concatenate img features and predict the original word based on the shifted input!
        self.sentences = sentences # a list of sentences
        self.images = images
        self.images_idx = images_idx
        self.max_len = max_len

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self,idx):
        img = self.images[self.images_idx[idx]].squeeze(0)
        # Pad to max len 
        sentence = torch.zeros((1,self.max_len))
        sentence[0,:len(self.sentences[idx])] = torch.Tensor(self.sentences[idx])
    
        return img, sentence
    
def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (image, caption).

    Args:
        data: list of tuple (image, caption). 
            - image: torch tensor of shape (4096).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 4096).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    # Sort a data list by caption length (descending order).
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions = zip(*data)
    
    # Merge images (from tuple of 3D tensor to 4D tensor).
    images = torch.stack(images, 0)

    # Get batch size 
    N = len(captions)
    LEN = captions[0].shape[1]
    # Create targets tensor 
    targets = torch.zeros((N, LEN),dtype=torch.long) 

    # Create target mask 
    # Shape must be (N, sentence_length, sentence_length)
    trg_masks = torch.tril(torch.ones((LEN,LEN),dtype=torch.int8).expand(N,1,LEN,LEN))

    for i, cap in enumerate(captions):
        targets[i] = cap
        pad_mask = (targets[i]!=0).unsqueeze(-2)
        trg_masks[i,0,:,:] = pad_mask & trg_masks[i,0,:,:]

    if torch.cuda.is_available():
        images = images.cuda()
        targets = targets.cuda()
        trg_masks = trg_masks.cuda()
        
    return images, targets, trg_masks