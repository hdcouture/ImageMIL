import sys
import os
from glob import glob
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser( description='Setup BreaKHis dataset.' )

    parser.add_argument('--in_dir', '-i', required=True, help='input directory' )
    parser.add_argument('--out_dir', '-o', required=True, help='output directory' )
    parser.add_argument('--mag', required=True, help='magnification' )
    args = parser.parse_args()
    src_dir = args.in_dir
    if len(src_dir) > 1 and src_dir[-1] != '/':
        src_dir += '/'
    out_dir = args.out_dir
    if len(out_dir) > 1 and out_dir[-1] != '/':
        out_dir += '/'
    magnification = args.mag

    root_dir = './BreaKHis_v1/histology_slides/breast'
    srcfiles = {'DC': 'malignant/SOB/ductal_carcinoma/%s/%sX/%s',
                'LC': 'malignant/SOB/lobular_carcinoma/%s/%sX/%s',
                'MC': 'malignant/SOB/mucinous_carcinoma/%s/%sX/%s',
                'PC': 'malignant/SOB/papillary_carcinoma/%s/%sX/%s',
                'A': 'benign/SOB/adenosis/%s/%sX/%s',
                'F': 'benign/SOB/fibroadenoma/%s/%sX/%s',
                'PT': 'benign/SOB/phyllodes_tumor/%s/%sX/%s',
                'TA': 'benign/SOB/tubular_adenoma/%s/%sX/%s'}
    sub_dir = 'histology_slides/breast/'

    if not os.path.exists(out_dir):
        print('Creating '+out_dir)
        os.makedirs(out_dir)
        
    for nfold in range(5):
        # image list
        db = open(src_dir+'dsfold%s.txt' % (nfold+1))

        print( 'Creating ' + out_dir + 'fold%s.txt' % nfold )
        fd_fold = open( out_dir + 'fold%s.txt' % nfold, 'w' )

        for row in db.readlines():
            columns = row.split('|')
            imgname = columns[0]
            mag = columns[1]  # 40, 100, 200, or 400
            grp = columns[3].strip()  # train or test

            s = imgname.split('-')
            name = s[0] + '-' + s[1] + '-' + s[2] + '-' + s[4].split('.')[0]

            if mag != magnification:
                continue

            if grp == 'train':
                fd_fold.write(name+',train\n')
            else:
                fd_fold.write(name+',test\n')

        fd_fold.close()

    db.close()

    print('Creating '+out_dir + 'sample_images.csv')
    fd_images = open( out_dir + 'sample_images.csv', 'w' )
    
    for k,v in srcfiles.items():
        for fn in glob( src_dir + sub_dir+ v.replace('%sX',magnification+'X').replace('%s','*') + '.png' ):
            imgname = fn.split('/')[-1]
            s = imgname.split('-')
            name = s[0] + '-' + s[1] + '-' + s[2] + '-' + s[4].split('.')[0]
            tumor_type = imgname.split('-')[0].split('_')[-1]

            fn = fn[fn.find(src_dir)+len(src_dir):]
            fn = fn[fn.find(sub_dir)+len(sub_dir):]

            fd_images.write( name+','+fn+'\n' )

    fd_images.close()
      
    print('Creating '+out_dir + 'labels.csv')
    fd_class = open( out_dir + 'labels.csv', 'w' )
    fd_class.write( 'sample,tumor,tumor_type,benign_type,malignant_type\n' )
    
    for k,v in srcfiles.items():
        for fn in glob( src_dir + sub_dir+ v.replace('%sX',magnification+'X').replace('%s','*') + '.png' ):
            imgname = fn.split('/')[-1]
            s = imgname.split('-')
            name = s[0] + '-' + s[1] + '-' + s[2] + '-' + s[4].split('.')[0]
            tumor_type = imgname.split('-')[0].split('_')[-1]

            if 'malignant' in fn:
                tumor = 'M'
                benign_type = ''
                malignant_type = tumor_type
            else:
                tumor = 'B'
                benign_type = tumor_type
                malignant_type = ''
            fd_class.write( '%s,%s,%s,%s,%s\n' % ( name, tumor, tumor_type, benign_type, malignant_type ) )

    fd_class.close()
                
