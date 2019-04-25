import os
import sys
import argparse
import numpy as np
import skimage.io

from keras.preprocessing import image
from keras.models import Model
from keras.layers import Input
from keras.layers.pooling import GlobalAveragePooling2D, AveragePooling2D
from keras.utils import print_summary

import util

if __name__ == "__main__":

    parser = argparse.ArgumentParser( description='Compute CNN features.' )
    parser.add_argument('--in_dir', '-i', required=True, help='input directory' )
    parser.add_argument('--out_dir', '-o', required=True, help='output directory' )
    parser.add_argument('--model', '-m', required=True, help='CNN model' )
    parser.add_argument('--layer', '-l', help='CNN layer' )
    parser.add_argument('--list-layers', action='store_true', help='list CNN layer names' )
    parser.add_argument('--instance-size', help='instance size' )
    parser.add_argument('--instance-stride', help='instance stride' )
    parser.add_argument('--pool-size', '-p', help='mean pooling size' )
    parser.add_argument('--mask', action='store_true', help='use mask' )
    args = parser.parse_args()
    src_dir = args.in_dir
    if len(src_dir) > 1 and src_dir[-1] != '/':
        src_dir += '/'
    out_dir = args.out_dir
    if len(out_dir) > 1 and out_dir[-1] != '/':
        out_dir += '/'
    model_name = args.model
    list_layers = args.list_layers
    layer = args.layer
    instance_size = args.instance_size
    instance_stride = args.instance_stride
    pool_size = args.pool_size
    use_mask = args.mask

    # load filenames and labels
    image_list = util.load_image_list( out_dir )
    if use_mask:
        mask_list = util.load_mask_list( out_dir )
    else:
        mask_list = [None]*len(image_list)

    # create model
    max_dim = None
    input_tensor = Input(shape=(max_dim,max_dim,3))
    if model_name.lower() == 'resnet50':
        from keras.applications.resnet50 import ResNet50
        from keras.applications.resnet50 import preprocess_input
        base_model = ResNet50(input_shape=(max_dim,max_dim,3),include_top=False,weights='imagenet')
    elif model_name.lower() == 'vgg16':
        from keras.applications.vgg16 import VGG16
        from keras.applications.vgg16 import preprocess_input
        base_model = VGG16(input_shape=(max_dim,max_dim,3),include_top=False,weights='imagenet')
    elif model_name.lower() == 'inceptionv3':
        from keras.applications.inception_v3 import InceptionV3
        from keras.applications.inception_v3 import preprocess_input
        base_model = InceptionV3(input_tensor=input_tensor,weights='imagenet')
    elif model_name.lower() == 'inceptionrenetv2':
        from keras.applications.inception_resnet_v2 import InceptionResNetV2
        from keras.applications.inception_resnet_v2 import preprocess_input
        base_model = InceptionResNetV2(input_tensor=input_tensor,weights='imagenet')
    elif model_name.lower() == 'xception':
        from keras.applications.Xception import Xception
        from keras.applications.Xception import preprocess_input
        base_model = Xception(input_tensor=input_tensor,weights='imagenet')
    else:
        print('Error: unsupported model')
        sys.exit(1)

    if list_layers:
        print_summary(base_model)
        sys.exit(0)

    x = base_model.get_layer(layer).output

    # for creating instances
    if pool_size is None and instance_size is None:
        x = GlobalAveragePooling2D(name='avgpool')(x)
    elif pool_size is not None:
        p = int(pool_size)
        if p > 0:
            x = AveragePooling2D((p,p),name='avgpool')(x)
    elif instance_size is not None:
        size = int(instance_size)
        if instance_stride is not None:
            stride = int(instance_stride)
        else:
            stride = size
        x = GlobalAveragePooling2D(name='avgpool')(x)
    model = Model( inputs=base_model.input, outputs=x )

    for img_fn,mask_fn in zip(image_list,mask_list):
        img = image.load_img( src_dir+img_fn )
        x = image.img_to_array( img )
        x = np.expand_dims( x, axis=0 )
        x = preprocess_input( x )

        if mask_fn:
            mask = skimage.io.imread( src_dir+mask_fn ).astype('float32')
            mask /= mask.max()

        if instance_size is not None:
            feat = []
            bs = 16
            x_batch = []
            for r in range(0,x.shape[1],stride):
                for c in range(0,x.shape[2],stride):
                    x_inst = x[:,r:min(r+size,x.shape[1]),c:min(c+size,x.shape[2]),:]
                    if use_mask:
                        # check mask
                        foreground = mask[r:min(r+size,x.shape[1]),c:min(c+size,x.shape[2])].mean()
                        if foreground < 0.5:
                            continue
                    if len(x_batch) >= bs or ( len(x_batch) > 0 and x_inst.shape != x_batch[0].shape ):
                        # process a batch
                        if len(x_batch) > 1:
                            x_batch = np.concatenate(x_batch,axis=0)
                        else:
                            x_batch = x_batch[0]
                        feat_batch = model.predict(x_batch)
                        feat.append( feat_batch )
                        x_batch = []
                    x_batch.append( x_inst )
            if len(x_batch) > 0:
                # process last batch
                if len(x_batch) > 1:
                    x_batch = np.concatenate(x_batch,axis=0)
                else:
                    x_batch = x_batch[0]
                feat_batch = model.predict(x_batch)
                feat.append( feat_batch )
            if len(feat) > 0:
                feat = np.concatenate( feat, axis=0 )
                feat = [ feat[r,:] for r in range(feat.shape[0]) ]
        else:
            p = model.predict(x)
            if len(p.shape) > 2:
                feat = [ p[:,r,c,:].squeeze() for r in range(p.shape[1]) for c in range(p.shape[2]) ]
            else:
                feat = [ p.squeeze() ]
        if len(feat) > 0:
            print('%d x %d' % (len(feat),feat[0].shape[0]))
        else:
            print('no instances')
            
        feat_fn = out_dir+img_fn[:img_fn.rfind('.')]+'_'+model_name+'-'+layer
        if pool_size is not None:
            feat_fn += '_p'+str(pool_size)
        if instance_size is not None:
            feat_fn += '_i'+str(instance_size)
            if instance_stride is not None:
                feat_fn += '-'+str(instance_stride)
        feat_fn += '.npy'
        print('Saving '+feat_fn)

        if not os.path.exists( os.path.dirname(feat_fn) ):
            os.makedirs( os.path.dirname(feat_fn) )
        np.save(feat_fn,feat)

