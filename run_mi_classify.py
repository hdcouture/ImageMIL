import os
import sys
import argparse
import numpy as np
import sklearn.model_selection
import sklearn.metrics

import util
from linear_classifier import LinearClassifier
from sil import SIL

class ResultsReport:
    def __init__(self,label_names=None):
        self.res = {}
        self.label_names = label_names
    def add(self,metric,result):
        if metric not in self.res:
            self.res[metric] = []
        self.res[metric].append( result )
    def print_summary(self,metric=None):
        if metric is None:
            for metric in sorted(self.res.keys()):
                if metric != 'confusion':
                    self.print_summary(metric)
            self.print_summary('confusion')
            return
        if metric != 'confusion':
            mean = np.mean(self.res[metric])
            std = np.std(self.res[metric])
            ste = std/np.sqrt(len(self.res[metric])-1)
            print('%s %f %f %f' % (metric,mean,std,ste) )
        else:
            print('confusion')
            print(('%s '*len(self.label_names))%tuple(self.label_names))
            print(sum(self.res['confusion']))

if __name__ == '__main__':

    parser = argparse.ArgumentParser( description='Compute CNN features.' )
    parser.add_argument('--out_dir', '-o', required=True, help='output directory' )
    parser.add_argument('--model', '-m', required=True, help='CNN model' )
    parser.add_argument('--layer', '-l', required=True, help='CNN layer' )
    parser.add_argument('--instance-size', help='instance size' )
    parser.add_argument('--instance-stride', help='instance stride' )
    parser.add_argument('--pool-size', '-p', help='mean pooling size' )
    parser.add_argument('--cat', help='label categories to train (comma separated); default: all' )
    parser.add_argument('--calibrate', action='store_true', help='calibrate classifier' )
    parser.add_argument('--metric', help='metric to optimize during parameter search (accuracy, balanced_accuracy, roc_auc); default: accuracy' )
    parser.add_argument('--classifier', '-c', help='classifier (svm or logistic); default: all' )
    parser.add_argument('--kernel', help='SVM kernel; default: linear' )
    parser.add_argument('--mi', help='MI type (none, median, quantile); default: none (compute mean across images)' )
    parser.add_argument('--quantiles', '-q', help='Number of quantiles; default: 16' )
    parser.add_argument('--sample-weight', help='Weight samples by classification category and this one' )
    parser.add_argument('--group', help='Class groups for reporting results' )
    parser.add_argument('--cv-fold-files', help='cross-validation fold files' )
    parser.add_argument('--cv-folds', help='cross-validation folds' )
    parser.add_argument('--cv-lno', help='cross-validation leave n out' )
    parser.add_argument('--n-jobs', help='number of parallel threads' )
    args = parser.parse_args()
    out_dir = args.out_dir
    if len(out_dir) > 1 and out_dir[-1] != '/':
        out_dir += '/'
    model_name = args.model
    layer = args.layer
    instance_size = args.instance_size
    instance_stride = args.instance_stride
    pool_size = args.pool_size
    categories = args.cat
    metric = args.metric
    calibrate = args.calibrate
    classifier = args.classifier
    kernel = args.kernel
    mi_type = args.mi
    quantiles = args.quantiles
    sample_weight = args.sample_weight
    group = args.group
    cv_fold_files = args.cv_fold_files
    cv_folds = args.cv_folds
    cv_lno = args.cv_lno
    n_jobs = args.n_jobs

    if calibrate is None:
        calibrate = False
    else:
        calibrate = bool(calibrate)
        print(calibrate)

    if n_jobs is not None:
        n_jobs = int(n_jobs)

    # load filenames and labels
    sample_images = util.load_sample_images( out_dir )
    samples,cats,labels = util.load_labels( out_dir )

    if sample_weight is not None:
        # get labels for sample_weight category
        c = np.where(cats==sample_weight)[0][0]
        ln = np.unique([l[c] for l in labels])
        ln.sort()
        ln = list(ln)
        if '' in ln:
            del ln[ln.index('')]
        label_names_sw = ln
        labels_sw = np.array([ ln.index(l) if l in ln else -1 for l in labels[:,c] ])
    if group is not None:
        # get labels for group category
        if group == sample_weight:
            label_names_group = label_names_sw
            labels_group = labels_sw
        else:
            c = np.where(cats==group)[0][0]
            ln = np.unique([l[c] for l in labels])
            ln.sort()
            ln = list(ln)
            if '' in ln:
                del ln[ln.index('')]
            label_names_group = ln
            labels_group = np.array([ ln.index(l) if l in ln else -1 for l in labels[:,c] ])
    if categories is None:
        # get labels for list of categories
        label_names = []
        new_labels = np.zeros(labels.shape,dtype='int')
        for c,cat in enumerate(cats):
            ln = np.unique([l[c] for l in labels])
            ln.sort()
            ln = list(ln)
            label_names.append( ln )
            new_labels[:,c] = [ ln.index(l) for l in labels[:,c] ]
        labels = new_labels
    else:
        # get labels for all categories
        label_names = []
        categories = categories.split(',')
        new_labels = np.zeros((labels.shape[0],len(categories)),dtype='int')
        for i,cat in enumerate(categories):
            c = np.where(cats==cat)[0][0]
            ln = np.unique([l[c] for l in labels])
            ln.sort()
            ln = list(ln)
            if '' in ln:
                del ln[ln.index('')]
            label_names.append( ln )
            new_labels[:,i] = np.array([ ln.index(l) if l in ln else -1 for l in labels[:,c] ])
        labels = new_labels
        cats = categories
        
    # read in CNN features
    feats = {}
    for sample,imagelist in sample_images.items():
        feats[sample] = []
        for fn in imagelist:
            feat_fn = out_dir+fn[:fn.rfind('.')]+'_'+model_name+'-'+layer
            if pool_size is not None:
                feat_fn += '_p'+str(pool_size)
            if instance_size is not None:
                feat_fn += '_i'+str(instance_size)
            if instance_stride is not None:
                feat_fn += '-'+str(instance_stride)
            feat_fn += '.npy'
            feat = np.load(feat_fn)
            if len(feat) == 0:
                continue
            feats[sample].append( feat )

        print('%s %d'%(sample,len(feats[sample])))
        feats[sample] = np.concatenate(feats[sample],axis=0)
        if len(feats[sample].shape) == 1:
            feats[sample] = feats[sample].reshape((1,len(feats[sample])))
            
        # compute mean if needed
        if mi_type is None or mi_type.lower() == 'none':
            if len(feats[sample].shape) > 1:
                feats[sample] = feats[sample].mean(axis=0)

    # build train/test sets
    if cv_fold_files is not None:
        idx_train_test = util.load_cv_files( out_dir, samples, cv_fold_files )
    elif cv_folds is not None or cv_lno is not None:
        if cv_folds is not None:
            cv_folds = int(cv_folds)
        else:
            cv_lno = int(cv_lno)
            if cv_folds is None:
                cv_folds = len(samples) // cv_lno
        idx = np.arange(len(samples))
        if len(label_names) == 1:
            if cv_lno == 1:
                skf = sklearn.model_selection.LeaveOneOut()
            else:
                skf = sklearn.model_selection.StratifiedKFold( n_splits=cv_folds, shuffle=True )
            idx_train_test = list(skf.split(idx,labels[:,0]))
        else:
            # merge label categories to do stratified folds
            skf = sklearn.model_selection.StratifiedKFold( n_splits=cv_folds, shuffle=True )
            la_all = np.array(labels[:,0])
            p = 1
            for i in range(labels.shape[1]):
                la_all += labels[:,i] * p
                p *= len(label_names[i])
            idx_train_test = list(skf.split(idx,la_all))
    else:
        print('Error: train/test split not specified')
        sys.exit(1)

    options = {}
    if kernel is not None:
        options['kernel'] = kernel
    else:
        options['kernel'] = 'linear'
    if classifier is not None:
        options['classifier'] = classifier
    if mi_type is not None:
        options['predict_type'] = mi_type
    if metric is not None:
        options['metric'] = metric
                        
    for c,cat_name in enumerate(cats):
        print(cat_name)
        res = ResultsReport(label_names[c])
        nfolds = len(idx_train_test)
        for f,(idx_train,idx_test) in enumerate(idx_train_test):
            print('Fold '+str(f+1)+'/'+str(len(idx_train_test)))
            idx_train = idx_train[np.where(labels[idx_train,c]!=-1)[0]]
            idx_test = idx_test[np.where(labels[idx_test,c]!=-1)[0]]
            X_train = [ feats[samples[i]] for i in idx_train ]
            y_train = labels[idx_train,c]
            X_test = [ feats[samples[i]] for i in idx_test ]
            y_test = labels[idx_test,c]

            if sample_weight is not None:
                # figure out sample weights
                print('Weighting by '+sample_weight)
                # discard samples missing a label for sample_weight category
                idx_train = idx_train[np.where(labels_sw[idx_train]!=-1)[0]]
                X_train = [ feats[samples[i]] for i in idx_train ]
                
                y_train = labels[idx_train,c]
                y_sw = y_train + len(label_names[c])*labels_sw[idx_train]

                uniq = np.unique(y_sw).tolist()
                counts = np.array([ (y_sw==l).sum() for l in uniq ])
                counts = counts.sum().astype(float) / ( counts * len(counts) )
                sw = np.array([ counts[uniq.index(y)] for y in y_sw ])
            else:
                sw = None

            if mi_type is None:
                model = LinearClassifier( n_jobs=n_jobs, **options )
                model.fit( X_train, y_train, calibrate=calibrate, param_search=True, sample_weight=sw )
            elif mi_type in ['median','max']:
                model = SIL( n_jobs=n_jobs, **options )
                model.fit( X_train, y_train, calibrate=calibrate, param_search=True, sample_weight=sw )
            elif mi_type == 'quantile':
                if quantiles is not None:
                    options['quantiles'] = int(quantiles)
                model = SIL( n_jobs=n_jobs, **options )
                model.fit( X_train, y_train, calibrate=calibrate, param_search=True, sample_weight=sw )
                
            p_predict = model.predict( X_test )
            y_predict = np.argmax(p_predict,axis=1)
            acc = sklearn.metrics.accuracy_score( y_test, y_predict )
            if len(y_test) == 1:
                auc = 0.0
            elif len(np.unique(y_train)) == 2:
                auc = sklearn.metrics.roc_auc_score( y_test, p_predict[:,1] )
            else:
                auc = 0.0
                for i in range(p_predict.shape[1]):
                    auc += sklearn.metrics.roc_auc_score( y_test==i, p_predict[:,i] )
                auc /= p_predict.shape[1]
            kappa = sklearn.metrics.cohen_kappa_score( y_test, y_predict )
            classes = np.unique(y_train)
            np.sort(classes)
            confusion = sklearn.metrics.confusion_matrix( y_test, y_predict, labels=classes )
            res.add('acc',acc)
            res.add('auc',auc)
            res.add('kappa',kappa)
            if len(label_names[c]) == 2:
                res.add('sensitivity', float( np.logical_and(y_test==1, y_predict==y_test).sum() ) / (y_test==1).sum() )
                res.add('specificity', float( np.logical_and(y_test!=1, y_predict==y_test).sum() ) / (y_test!=1).sum() )
            res.add('confusion',confusion)

            print('accuracy %f auc %f' % (acc,auc))
            print(confusion)

            if group is not None:
                # within group class metrics
                l_group = labels_group[idx_test]
                uniq = np.unique(l_group)
                uniq.sort()
                for u in uniq:
                    if u == -1:
                        continue
                    idx = (l_group==u)

                    group_name = '(%s=%s)'%(group,label_names_group[u])
                    res.add('accuracy '+group_name,sklearn.metrics.accuracy_score( y_test[idx], y_predict[idx] ))
                    if len(np.unique(y_train)) == 2:
                        if (y_test[idx]==0).sum() == 0 or (y_test[idx]==1).sum() == 0:
                            auc = 0
                        else:
                            auc = sklearn.metrics.roc_auc_score( y_test[idx], p_predict[idx,1] )
                    else:
                        auc = 0.0
                        for i in range(p_predict.shape[1]):
                            auc += sklearn.metrics.roc_auc_score( y_test[idx]==i, p_predict[idx,i] )
                        auc /= p_predict.shape[1]
                    res.add('auc '+group_name,auc)
                    res.add('kappa '+group_name,sklearn.metrics.cohen_kappa_score( y_test[idx], y_predict[idx] ) )
                    if len(label_names[c]) == 2:
                        res.add('sensitivity '+group_name,float( np.logical_and(y_test[idx]==1, y_predict[idx]==y_test[idx]).sum() ) / (y_test[idx]==1).sum() )
                        res.add('specificity '+group_name,float( np.logical_and(y_test[idx]!=1, y_predict[idx]==y_test[idx]).sum() ) / (y_test[idx]!=1).sum() )
            
        print('Cross-validation results')
        res.print_summary()
