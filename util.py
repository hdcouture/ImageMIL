import csv
from glob import glob
import numpy as np

def load_image_list( out_dir ):

    img_files = []
    fd = open( out_dir + 'sample_images.csv' )
    for line in fd:
        files = [ fn.strip() for fn in line.split(',')[1:] if fn.strip() != '' ]
        img_files.extend( files )
    return img_files

def load_mask_list( out_dir ):

    mask_files = []
    fd = open( out_dir + 'sample_masks.csv' )
    for line in fd:
        mask_files.extend( [ fn.strip() for fn in line.split(',')[1:] if fn.strip() != '' ] )
    return mask_files

def load_sample_images( out_dir ):

    samples = {}
    fd = open( out_dir + 'sample_images.csv' )
    for line in fd:
        line = line.split(',')
        samples[line[0]] = [ fn.strip() for fn in line[1:] if fn.strip() != '' ]
    return samples
    
def load_labels( out_dir ):

    samples = []
    labels = []
    #d = np.loadtxt( out_dir+'labels.csv', dtype=str, delimiter=',' )
    #d = np.array(d)
    d = []
    with open( out_dir+'labels.csv', 'r' ) as csvfile:
        reader = csv.reader( csvfile )
        for row in reader:
            d.append( row )
    #print([len(di) for di in d])
    d = np.vstack(d)
    samples = d[1:,0]
    cats = d[0,1:]
    labels = d[1:,1:]
    return samples,cats,labels

def load_cv_files( out_dir, samples, cv_fold_files ):
        
    cv_files = sorted(list(glob( out_dir + cv_fold_files )))
    idx_train_test = []
    for fn in cv_files:
        print(fn)
        f = np.loadtxt( fn, dtype=str, delimiter=',' )
        idx_train = np.where(f[:,1]=='train')[0]
        idx_test = np.where(f[:,1]=='test')[0]
        name_train = f[idx_train,0]
        name_test = f[idx_test,0]
        idx_train = np.array([ np.where(samples==name)[0] for name in name_train ]).flatten()
        idx_test = np.array([ np.where(samples==name)[0] for name in name_test ]).flatten()
        idx_train_test.append( [idx_train,idx_test] )
    return idx_train_test
        
