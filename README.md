# ImageMIL - Multiple Instance Learning on Large Images

These methods were developed for classifying larger images with intra-image heterogeneity.  The class of each image is not dictated by some small to-be-identified region but by the overall appearance - perhaps the average class of smaller regions or some more complex relationship.

Multiple Instance (MI) terminology: All images from a particular sample are referred to as a "bag" and each image region is called an "instance."  A bag can have many instances. Labels are given at the bag level, not the instance level, making this a weakly supervised learning problem.

With the standard MI assumption, a sample is classiifed as positive if and only if at least one of its instances is positive.  This asymmetric relationship works well for problems such a cancer diagnosis where the presence of even a small region of tumor should produce a prediction of cancer.  For many other applications, it is more appropriate to assign a label to an image based on the properties of multiple regions.  The code tackles such a challenge.

The methods implemented here include those discussed in the following two publications:

H. D. Couture, L. Williams, J. Geradts, S. Nyante, E. Butler, J. Marron, C. Perou, M. Troester, and M. Niethammer, “Image analysis with deep learning to predict breast cancer grade, ER status, histologic subtype, and intrinsic subtype,” npj Breast Cancer, 2018.

H. D. Couture, Discriminative Representations for Heterogeneous Images and Multimodal Data. PhD thesis, Department of Computer Science, University of North Carolina at Chapel Hill, Chapel Hill, NC, 2019.

In the latter, it is the SIL-quantile method of Chapter 2.

This code is not the original used in these publications but an upgraded version to work with the latest version of Keras and other libraries.  Tested with Python 3.7.3, Keras 2.2.4, sklearn 0.20.3, and skimage 0.15.0.

## Setup

Basic installation requires a number of python packages, which are most easily installed with conda:

```
conda install -c conda-forge numpy scipy keras scikit-learn scikit-image
```

## Data Setup

The above referenced publications used data from the [Carolina Breast Cancer Study](http://cbcs.web.unc.edu/for-researchers/).  You may apply for access to this data set.

This code was also setup to run on the [BreaKHis](https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/) data set for breast cancer histopathological classification.  A setup file for this data set is included (setup_breakhis.py) that creates the required input files for this code.

Running this code requires two files: labels.csv and sample_images.csv.

labels.csv should use the following format for up to N samples and K classification tasks:
```
sample,class1,class2,...,classK
sample1,label11,label12,...,label1K
...
sampleN,labelN1,labelN2,...,labelNK
```

Each class can be binary or multi-class.  Any string or number can be used to identify the classes.

sample_images.csv allows one or more image files to be specified for each sample:
```
sample1,image11,image12,...,image1M
...
sampleN,imageN1,imageN2,...,imageNM
```

Each sample may have a different number of associated images.

If a specific train/test split is needed, a file or files may be provided in the following format:
```
sample1,train
sample2,train
sample3,test
...
sampleN,train
```

## Example Usage for BreaKHis

```
python setup_breakhis.py -i BreaKHis_v1/ -o BreaKHis200/ --mag 200
python run_cnn_features.py -i BreaKHis_v1/histology_slides/breast/ -o BreaKHis200/ -m vgg16 -l block4_pool --pool-size 5
python run_mi_classify.py -o BreaKHis200/ -m vgg16 -l block4_pool --cat tumor --cv-fold-files fold* --pool-size 5 --mi median
```

## Example Usage for CBCS

```
python setup_cbcs.py -i CBCS/images/ -o CBCS_out/ --spreadsheet CBCS.csv
python run_cnn_features.py -i CBCS/images/ -o CBCS_out/ -m vgg16 -l block4_pool --instance-size 800 --instance-stride 400
python run_mi_classify.py -o CBCS_out/ -m vgg16 -l block4_pool --cat grade1vs3 --cv-folds 5 --instance-size 800 --instance-stride 400 --mi quantile
python run_mi_classify.py -o CBCS_out/ -m vgg16 -l block4_pool --cat BasalvsNonBasal,er,ductal_lobular,ror-high --sample-weight grade12vs3 --group grade12vs3 --cv-folds 5 --instance-size 800 --instance-stride 400 --mi quantile
```

