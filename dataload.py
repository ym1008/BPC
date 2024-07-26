import os
import csv
import random
import kaldiio
import augment
import torch
import numpy as np
import soundfile as sf
import scipy as sp
from torch.utils import data


def enumerateLabels(labels, ctcloss, nsn = False):

    key_class = ['SIL', 'SPN', 'AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'B', 'CH', 'D', 'DH', 'EH', 'ER', 'EY', 'F', 'G', 'HH', 'IH', 'IY', 'JH', 'K', 'L', 'M', 'N', 'NG', 'OW', 'OY', 'P', 'R', 'S', 'SH', 'T', 'TH', 'UH', 'UW', 'V', 'W', 'Y', 'Z', 'ZH']
    if nsn:
        key_class.append('NSN')

    value_class = range(len(key_class))
    dict_class = dict(zip(key_class,value_class))
        
    keys = ['SIL\n', 'SIL', 'SIL_B', 'SIL_E', 'SIL_I', 'SIL_S', 'SPN', 'SPN_B', 'SPN_E', 'SPN_I', 'SPN_S', 'AA_B', 'AA_E', 'AA_I', 'AA_S', 'AA0_B', 'AA0_E', 'AA0_I', 'AA0_S', 'AA1_B', 'AA1_E', 'AA1_I', 'AA1_S', 'AA2_B', 'AA2_E', 'AA2_I', 'AA2_S', 'AE_B', 'AE_E', 'AE_I', 'AE_S', 'AE0_B', 'AE0_E', 'AE0_I', 'AE0_S', 'AE1_B', 'AE1_E', 'AE1_I', 'AE1_S', 'AE2_B', 'AE2_E', 'AE2_I', 'AE2_S', 'AH_B', 'AH_E', 'AH_I', 'AH_S', 'AH0_B', 'AH0_E', 'AH0_I', 'AH0_S', 'AH1_B', 'AH1_E', 'AH1_I', 'AH1_S', 'AH2_B', 'AH2_E', 'AH2_I', 'AH2_S', 'AO_B', 'AO_E', 'AO_I', 'AO_S', 'AO0_B', 'AO0_E', 'AO0_I', 'AO0_S', 'AO1_B', 'AO1_E', 'AO1_I', 'AO1_S', 'AO2_B', 'AO2_E', 'AO2_I', 'AO2_S', 'AW_B', 'AW_E', 'AW_I', 'AW_S', 'AW0_B', 'AW0_E', 'AW0_I', 'AW0_S', 'AW1_B', 'AW1_E', 'AW1_I', 'AW1_S', 'AW2_B', 'AW2_E', 'AW2_I', 'AW2_S', 'AY_B', 'AY_E', 'AY_I', 'AY_S', 'AY0_B', 'AY0_E', 'AY0_I', 'AY0_S', 'AY1_B', 'AY1_E', 'AY1_I', 'AY1_S', 'AY2_B', 'AY2_E', 'AY2_I', 'AY2_S', 'B_B', 'B_E', 'B_I', 'B_S', 'CH_B', 'CH_E', 'CH_I', 'CH_S', 'D_B', 'D_E', 'D_I', 'D_S', 'DH_B', 'DH_E', 'DH_I', 'DH_S', 'EH_B', 'EH_E', 'EH_I', 'EH_S', 'EH0_B', 'EH0_E', 'EH0_I', 'EH0_S', 'EH1_B', 'EH1_E', 'EH1_I', 'EH1_S', 'EH2_B', 'EH2_E', 'EH2_I', 'EH2_S', 'ER_B', 'ER_E', 'ER_I', 'ER_S', 'ER0_B', 'ER0_E', 'ER0_I', 'ER0_S', 'ER1_B', 'ER1_E', 'ER1_I', 'ER1_S', 'ER2_B', 'ER2_E', 'ER2_I', 'ER2_S', 'EY_B', 'EY_E', 'EY_I', 'EY_S', 'EY0_B', 'EY0_E', 'EY0_I', 'EY0_S', 'EY1_B', 'EY1_E', 'EY1_I', 'EY1_S', 'EY2_B', 'EY2_E', 'EY2_I', 'EY2_S', 'F_B', 'F_E', 'F_I', 'F_S', 'G_B', 'G_E', 'G_I', 'G_S', 'HH_B', 'HH_E', 'HH_I', 'HH_S', 'IH_B', 'IH_E', 'IH_I', 'IH_S', 'IH0_B', 'IH0_E', 'IH0_I', 'IH0_S', 'IH1_B', 'IH1_E', 'IH1_I', 'IH1_S', 'IH2_B', 'IH2_E', 'IH2_I', 'IH2_S', 'IY_B', 'IY_E', 'IY_I', 'IY_S', 'IY0_B', 'IY0_E', 'IY0_I', 'IY0_S', 'IY1_B', 'IY1_E', 'IY1_I', 'IY1_S', 'IY2_B', 'IY2_E', 'IY2_I', 'IY2_S', 'JH_B', 'JH_E', 'JH_I', 'JH_S', 'K_B', 'K_E', 'K_I', 'K_S', 'L_B', 'L_E', 'L_I', 'L_S', 'M_B', 'M_E', 'M_I', 'M_S', 'N_B', 'N_E', 'N_I', 'N_S', 'NG_B', 'NG_E', 'NG_I', 'NG_S', 'OW_B', 'OW_E', 'OW_I', 'OW_S', 'OW0_B', 'OW0_E', 'OW0_I', 'OW0_S', 'OW1_B', 'OW1_E', 'OW1_I', 'OW1_S', 'OW2_B', 'OW2_E', 'OW2_I', 'OW2_S', 'OY_B', 'OY_E', 'OY_I', 'OY_S', 'OY0_B', 'OY0_E', 'OY0_I', 'OY0_S', 'OY1_B', 'OY1_E', 'OY1_I', 'OY1_S', 'OY2_B', 'OY2_E', 'OY2_I', 'OY2_S', 'P_B', 'P_E', 'P_I', 'P_S', 'R_B', 'R_E', 'R_I', 'R_S', 'S_B', 'S_E', 'S_I', 'S_S', 'SH_B', 'SH_E', 'SH_I', 'SH_S', 'T_B', 'T_E', 'T_I', 'T_S', 'TH_B', 'TH_E', 'TH_I', 'TH_S', 'UH_B', 'UH_E', 'UH_I', 'UH_S', 'UH0_B', 'UH0_E', 'UH0_I', 'UH0_S', 'UH1_B', 'UH1_E', 'UH1_I', 'UH1_S', 'UH2_B', 'UH2_E', 'UH2_I', 'UH2_S', 'UW_B', 'UW_E', 'UW_I', 'UW_S', 'UW0_B', 'UW0_E', 'UW0_I', 'UW0_S', 'UW1_B', 'UW1_E', 'UW1_I', 'UW1_S', 'UW2_B', 'UW2_E', 'UW2_I', 'UW2_S', 'V_B', 'V_E', 'V_I', 'V_S', 'W_B', 'W_E', 'W_I', 'W_S', 'Y_B', 'Y_E', 'Y_I', 'Y_S', 'Z_B', 'Z_E', 'Z_I', 'Z_S', 'ZH_B', 'ZH_E', 'ZH_I', 'ZH_S']
    values = ['SIL', 'SIL', 'SIL', 'SIL', 'SIL', 'SIL', 'SPN', 'SPN', 'SPN', 'SPN', 'SPN', 'AA', 'AA', 'AA', 'AA', 'AA', 'AA', 'AA', 'AA', 'AA', 'AA', 'AA', 'AA', 'AA', 'AA', 'AA', 'AA', 'AE', 'AE', 'AE', 'AE', 'AE', 'AE', 'AE', 'AE', 'AE', 'AE', 'AE', 'AE', 'AE', 'AE', 'AE', 'AE', 'AH', 'AH', 'AH', 'AH', 'AH', 'AH', 'AH', 'AH', 'AH', 'AH', 'AH', 'AH', 'AH', 'AH', 'AH', 'AH', 'AO', 'AO', 'AO', 'AO', 'AO', 'AO', 'AO', 'AO', 'AO', 'AO', 'AO', 'AO', 'AO', 'AO', 'AO', 'AO', 'AW', 'AW', 'AW', 'AW', 'AW', 'AW', 'AW', 'AW', 'AW', 'AW', 'AW', 'AW', 'AW', 'AW', 'AW', 'AW', 'AY', 'AY', 'AY', 'AY', 'AY', 'AY', 'AY', 'AY', 'AY', 'AY', 'AY', 'AY', 'AY', 'AY', 'AY', 'AY', 'B', 'B', 'B', 'B', 'CH', 'CH', 'CH', 'CH', 'D', 'D', 'D', 'D', 'DH', 'DH', 'DH', 'DH', 'EH', 'EH', 'EH', 'EH', 'EH', 'EH', 'EH', 'EH', 'EH', 'EH', 'EH', 'EH', 'EH', 'EH', 'EH', 'EH', 'ER', 'ER', 'ER', 'ER', 'ER', 'ER', 'ER', 'ER', 'ER', 'ER', 'ER', 'ER', 'ER', 'ER', 'ER', 'ER', 'EY', 'EY', 'EY', 'EY', 'EY', 'EY', 'EY', 'EY', 'EY', 'EY', 'EY', 'EY', 'EY', 'EY', 'EY', 'EY', 'F', 'F', 'F', 'F', 'G', 'G', 'G', 'G', 'HH', 'HH', 'HH', 'HH', 'IH', 'IH', 'IH', 'IH', 'IH', 'IH', 'IH', 'IH', 'IH', 'IH', 'IH', 'IH', 'IH', 'IH', 'IH', 'IH', 'IY', 'IY', 'IY', 'IY', 'IY', 'IY', 'IY', 'IY', 'IY', 'IY', 'IY', 'IY', 'IY', 'IY', 'IY', 'IY', 'JH', 'JH', 'JH', 'JH', 'K', 'K', 'K', 'K', 'L', 'L', 'L', 'L', 'M', 'M', 'M', 'M', 'N', 'N', 'N', 'N', 'NG', 'NG', 'NG', 'NG', 'OW', 'OW', 'OW', 'OW', 'OW', 'OW', 'OW', 'OW', 'OW', 'OW', 'OW', 'OW', 'OW', 'OW', 'OW', 'OW', 'OY', 'OY', 'OY', 'OY', 'OY', 'OY', 'OY', 'OY', 'OY', 'OY', 'OY', 'OY', 'OY', 'OY', 'OY', 'OY', 'P', 'P', 'P', 'P', 'R', 'R', 'R', 'R', 'S', 'S', 'S', 'S', 'SH', 'SH', 'SH', 'SH', 'T', 'T', 'T', 'T', 'TH', 'TH', 'TH', 'TH', 'UH', 'UH', 'UH', 'UH', 'UH', 'UH', 'UH', 'UH', 'UH', 'UH', 'UH', 'UH', 'UH', 'UH', 'UH', 'UH', 'UW', 'UW', 'UW', 'UW', 'UW', 'UW', 'UW', 'UW', 'UW', 'UW', 'UW', 'UW', 'UW', 'UW', 'UW', 'UW', 'V', 'V', 'V', 'V', 'W', 'W', 'W', 'W', 'Y', 'Y', 'Y', 'Y', 'Z', 'Z', 'Z', 'Z', 'ZH', 'ZH', 'ZH', 'ZH']
    if nsn:
        keys.append('NSN')
        keys.append('NSN_S')
        values.append('NSN')
        values.append('NSN')

    for i, value in enumerate(values):
        values[i] = dict_class[value]

    mapping = dict(zip(keys,values))

    if ctcloss:
        pho_pre = labels[0]
        class_labels=[mapping[pho_pre]]

        for pho in labels[1:]:
            if mapping[pho] == mapping[pho_pre]:
                s1 = pho_pre.split('_')
                s2 = pho.split('_')
                if len(s1) == 2:
                    s1 = s1[1]
                    s2 = s2[1]
                    if s1 == 'E' and (s2 == 'S' or s2 == 'B'):
                        class_labels.append(mapping[pho])
                        
                    if s1 == 'S' and s2 == 'B':
                        class_labels.append(mapping[pho])
            else:
                class_labels.append(mapping[pho])

            pho_pre = pho

        nlabels = len(class_labels)
        
        # pad out
        while len(class_labels) < len(labels):
            class_labels.append(class_labels[-1])
        labels = class_labels
    
    else:
        for i, pho in enumerate(labels):
            if pho in mapping:
                labels[i] = mapping[pho]
        nlabels = len(labels)

    return (np.array(nlabels), np.array(labels))



def apply_augmentation(data, augmentation, sample_size):

    data = augmentation(data)
    data = data[:, : sample_size]
    diff = sample_size - data.size(1)
                       
    if diff > 0:
        data = torch.cat([data, torch.tensor([[data[0, -diff]]])], dim = 1)

    return data



class LibriSpeech(data.Dataset):
   
    def __init__(self, filename, no_input_norm, sample_size, train, augment, label_fname = None, rfs = 400, stride = 160, ctcloss = False, nsn = False):
        
        with open(filename) as fd:
            reader = csv.reader(fd, delimiter='\t')
            header = next(reader)
            files = [[fname, tdim, label_location] for fname, tdim, label_location in reader]
        
        nsamples = len(files)
        files = np.array(files)
        
        self.tdim = files[:,1].astype('int')
        self.filenames = files[:,0].astype('string_')
        self.sample_size = sample_size
        self.train = train
        self.data_directory = header[0]
        self.nsamples = nsamples
        self.norm_per_sample = not no_input_norm
        self.augment_past = augment[0]
        self.augment_future = augment[1]
        

        # properties related to supervised downstream task
        self.rfs = rfs
        self.stride = stride
        self.label_fname = label_fname
        self.label_location = files[:,2].astype('int')
        self.ctcloss = ctcloss
        self.nsn = nsn
        

        
    def __getitem__(self, index):
        
        fname = self.filenames[index].decode('UTF-8')
        tdim = self.tdim[index]
        data, rate = sf.read(os.path.join(self.data_directory, fname), dtype='float32') 

        # get phone labels data
        if self.label_fname is not None:
            label_location = self.label_location[index]
      
            with open(self.label_fname) as f:
                tmp = f.seek(label_location)
                labels = f.readline()

            labels = labels.split('\n')[0]
            labels = labels.split(' ') 
            if labels[-1] == '\n':
                labels = labels[1:-1]
            else:
                labels = labels[1:]


            nframes = int(np.floor((self.sample_size - self.rfs) / self.stride + 1))
            ntotal = int(np.floor((tdim - self.rfs) / self.stride + 1))
            ntotal = ntotal - 1 if ntotal > nframes else ntotal
            scale = int(self.stride / 160)

            if self.train:
                idf = random.randint(0, ntotal - nframes) # random cropping 
                data = data[ idf * self.stride : idf * self.stride + self.sample_size ]
                labels = labels[idf * scale: (idf + nframes)*scale : scale] 
            else:
                data = data[:self.sample_size]
                labels = labels[: nframes * scale: scale]
         
            labels = enumerateLabels(labels, self.ctcloss, self.nsn)
        
        else:
            labels = np.array([])
            if self.train:
                idt = random.randint(0, tdim - self.sample_size) # random cropping 
                data = data[idt:idt + self.sample_size]
            else:
                data = data[:self.sample_size]



        # normalisation
        if self.norm_per_sample:
            data = (data - np.mean(data)) / np.sqrt(np.var(data) + 1e-5) 
        
        data = torch.from_numpy(data).unsqueeze(0)
                
        # augmentations
        past = apply_augmentation(data, self.augment_past, self.sample_size) if self.augment_past else data
        future = apply_augmentation(data, self.augment_future, self.sample_size) if self.augment_future else data


        return ((past, future), labels)


    def __len__(self):

        return self.nsamples




