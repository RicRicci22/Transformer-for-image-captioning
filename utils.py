def convert_to_words_ucm(input_file):
    with open(input_file,'r',encoding='utf-8') as file:
        content = file.read().lower()    
    sentences = dict()
    max_l = 0
    for row in content.split('\n'):
        pieces = row.strip().split(' ') 
        pieces.append('endseq')
        pieces.insert(1,'startseq')
        filename = int(pieces[0])
        del(pieces[0])
        try:
            sentences[filename].append(pieces)
        except:
            sentences[filename] = []
            sentences[filename].append(pieces)
        
        if(len(pieces))>max_l:
            max_l = len(pieces)

    return sentences,max_l

def create_lists_ucm_uav(train_filenames,val_filenames,test_filenames):
    train_ = []
    val_ = []
    test_ = []
    # Train
    with open(train_filenames,'r') as file:
        train = file.readlines()
    for line in train:
        train_.append(int(line.split('.')[0]))
    # Test
    with open(test_filenames,'r') as file:
        test = file.readlines()
    for line in test:
        test_.append(int(line.split('.')[0]))
    # Val 
    with open(val_filenames,'r') as file:
        val = file.readlines()
    for line in val:
        val_.append(int(line.split('.')[0]))
    
    return train_,val_,test_


def word_frequency_ucm_uav(text_in_words,min_word_frequency,test_sentences):
    word_freq = {}
    for question in text_in_words:
        for word in question:
            if(word in word_freq.keys()):
                word_freq[word] += 1
            else:
                word_freq[word] = 1

    ignored_words = set()
    for k, v in word_freq.items():
        if word_freq[k] <= min_word_frequency:
            ignored_words.add(k)
    print('Unique words before ignoring:', len(word_freq.keys()))
    print('Ignoring words with frequency <', min_word_frequency)
    words = [k for k in word_freq.keys() if k not in ignored_words]

    word_indices = dict((c, i+1) for i, c in enumerate(words))
    indices_word = dict((i+1, c) for i, c in enumerate(words))
    
    # Add unk token
    i = max(indices_word.keys())
    word_indices['unk'] = i+1
    indices_word[i+1] = 'unk'
    
    print('Unique words after ignoring:', len(indices_word))
    
    # Add unknown words from test sentences
    for image_sentences in test_sentences:
        for sentence in image_sentences:
            for word in sentence:
                try:
                    a = word_indices[word]
                except:
                    ignored_words.add(word)
    
    return word_indices,indices_word,ignored_words