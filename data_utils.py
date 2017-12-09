from sklearn.preprocessing import LabelBinarizer

import numpy as np
import json

alphabet = "abcdefghijklmnopqrstuvwxyz0123456789-,;.!?:'\"/\\|_@#$%^&*~`+-=<>()[]{}\n"

class YelpDataset():    
    
    def __init__(self, path):
        self.path = path

    def __len__(self):
        return len(self.dataset)
   
    def load(self):
        
        x, y = self.generate_data()
        
        print("X: {}".format(x.shape))
        print("Y: {}".format(y.shape))
        
        return x, y
    
    def generate_data(self):
        x = []
        y = []
        with open(self.path) as dfile:
            count = 0
            
            for line in dfile:
                review = json.loads(line)
                stars = review["stars"]
                text = review["text"]
                
                # Non neutral reviews
                if stars != 3:
                    clipped = self.clip_seq(list(text.lower()))
                    padded = self.pad_seq(clipped)
                    int_seq = self.str_to_int8(padded)
                    if stars == 1 or stars == 2:
                        x.append(int_seq)
                        y.append([1, 0])
                    elif stars == 4 or stars == 5:
                        x.append(int_seq)
                        y.append([0, 1])
                    count += 1
                    if count % 1000 == 0:
                        print("{} non-neutral instances processed".format(count))
                        break
        return np.array(x), np.array(y)


    def clip_seq(self, char_seq):
        if len(char_seq) > 1014:
            char_seq = char_seq[-1014:]
        return char_seq


    def pad_seq(self, char_seq, seq_length=1014, pad_char=" "):
        pad_width = seq_length - len(char_seq)
        padded_seq = char_seq + [pad_char] * pad_width
        return padded_seq


    def str_to_int8(self, char_seq):
        return np.array([alphabet.find(char) for char in char_seq], dtype=np.int8)


def one_hot_x(x, y, start_idx, end_idx):
    x_batch = x[start_idx:end_idx]
    y_batch = y[start_idx:end_idx]
    one_hot_batch = []
    
    binarizer = LabelBinarizer()
    binarizer.fit(range(len(alphabet)))
    
    for x in x_batch:
        one_hot_batch.append(binarizer.transform(x))
    one_hot_batch = np.array([one_hot_batch])
    x_batch = np.transpose(one_hot_batch, (1, 3, 2, 0))
    return x_batch, y_batch

def batch_iter(x, y, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    print ("Generating batch iterator ...")
    data_size = len(x)
    num_batches_per_epoch = int(data_size/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            x_shuff = x[shuffle_indices]
            y_shuff = y[shuffle_indices]
        else:
            x_shuff = x
            y_shuff = y
        for batch_num in range(num_batches_per_epoch):
            start_idx = batch_num * batch_size
            end_idx = min((batch_num + 1) * batch_size, data_size)
            x_batch, y_batch = one_hot_x(x_shuff, y_shuff, start_idx, end_idx)
            yield list(zip(x_batch, y_batch))