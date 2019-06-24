import numpy as np
import sklearn
import sklearn.model_selection

from linear_classifier import LinearClassifier
from linear_classifier import balanced_accuracy

def slices(groups):
    """
    Generate slices to select
    groups of the given sizes
    within a list/matrix
    """
    i = 0
    for group in groups:
        yield i, i + group
        i += group

class SIL(LinearClassifier):
    """Single-Instance Learning applied to MI data."""

    def __init__(self, classifier='svm', kernel='linear', C=None, p=3, gamma=1e0, T=None, predict_type='median', class_weight='balanced', calibrate=False, subset=None, quantiles=16, metric='acc', n_jobs=1, verbose=True):

        self.T = T
        self.predict_type = predict_type
        self.n_jobs = n_jobs
        self.classifier = classifier
        self.C = C
        self._model_agg = None
        self._calibrate = calibrate
        self.subset = subset
        self.quantiles = quantiles
        super(SIL, self).__init__(classifier=classifier,kernel=kernel,class_weight=class_weight,C=C,p=p,gamma=gamma,subset=subset,metric=metric,n_jobs=n_jobs,verbose=verbose)

    def fit(self, bags, y=None, sample_weight=None, param_search=False, calibrate=False):
        """Fit model."""

        if type(y[0]) is tuple:
            classes = [ yi[1].squeeze() for yi in y ]
            classes = np.hstack(classes)
            classes = np.array(classes).squeeze()
            y = [ yi[0] for yi in y ]
        else:
            classes = None

        # get into the format needed
        y = np.array(y)
        y = np.asmatrix(y).reshape((-1, 1)).astype(float)
        self._bags = [np.asmatrix(bag) for bag in bags]
        svm_X = np.vstack(self._bags)
        if classes is not None:
            svm_y = classes
        else:
            svm_y = np.vstack([float(cls) * np.matrix(np.ones((len(bag), 1)))
                               for bag, cls in zip(self._bags, y)])
            svm_y = np.array(svm_y).squeeze()

        if sample_weight is not None:
            # compute sample weights
            sample_weight2 = np.hstack(np.array( [ [1.0/len(bag)*sw]*len(bag) for bag,sw in zip(bags,sample_weight) ] )).squeeze()
            sample_weight2 = sample_weight2 * len(sample_weight2) / sample_weight2.sum()
        else:
            sample_weight2 = None
        y = np.array(y)

        if param_search:
            C,gamma = self.param_search(bags,y,svm_X,svm_y,C=self.C,gamma=self.gamma,sample_weight=sample_weight)
            self.C = C
            self.gamma = gamma

        if 'quantile' in self.predict_type:
            # set train/test splits for cross-validation
            bag_inst2idx = {}
            count = 0
            for b,bag in enumerate(bags):
                for i in range(len(bag)):
                    bag_inst2idx[b,i] = count
                    count += 1
            skf = sklearn.model_selection.StratifiedKFold( n_splits=5, shuffle=True )
            cv_split_bags = []
            cv_split_inst = []
            for train_idx, test_idx in skf.split(bags,y):
                cv_split_bags.append( (train_idx, test_idx) )
                train_idx_inst = []
                test_idx_inst = []
                for b in train_idx:
                    for i in range(len(bags[b])):
                        train_idx_inst.append( bag_inst2idx[b,i] )
                for b in test_idx:
                    for i in range(len(bags[b])):
                        test_idx_inst.append( bag_inst2idx[b,i] )
                cv_split_inst.append( (train_idx_inst, test_idx_inst) )
        else:
            cv_split_inst = None
            cv_split_bags = None

        super(SIL, self).fit(svm_X, svm_y,sample_weight=sample_weight2,calibrate=((param_search and self._calibrate) or calibrate),cv_split=cv_split_inst)

        if 'quantile' in self.predict_type:
            self.train_model_agg( bags, y, cv_split_bags=cv_split_bags, sample_weight=sample_weight )
        else:
            self._model_agg = None

        return self

    def train_model_agg( self, bags, y, cv_split_bags=None, sample_weight=None, param_search=True ):
        """Train instance aggregation function using quantile function."""

        # figure out number of quantiles and where to set them
        ninst = int( np.round( sum( [ len(bag) for bag in bags ] ) / float(len(bags)) ) )
        if self.quantiles is not None:
            nq = self.quantiles
        else:
            nq = 16
        if ninst <= nq:
            quantiles = np.linspace(0,100,ninst)
        else:
            quantiles = np.linspace(100.0/nq/2,100-100.0/nq/2,nq)

        p = []
        test_y = []
        if cv_split_bags is None:
            # train/test split
            skf = sklearn.model_selection.StratifiedKFold( n_splits=5, shuffle=True )
            cv_split_bags = list(skf.split(bags,y))

        # compute quantile function
        for f in range(5):
            train_idx,test_idx = cv_split_bags[f]
            for i in test_idx:
                pi = super(SIL,self).predict( bags[i], cv=f )
                if pi.shape[1] == 2:
                    q = np.percentile( pi[:,1], quantiles )
                else:
                    q = np.hstack( [ np.percentile( pi[:,c], quantiles ) for c in range(pi.shape[1]) ] )
                p.append( q )
                test_y.append( y[i] )
        p = np.vstack(p)
        test_y = np.array(test_y)

        # train model
        model_agg = LinearClassifier( classifier='svm' )
        self.C_agg,self.gamma_agg = model_agg.param_search( p, test_y, sample_weight=sample_weight, quick=False )
        model_agg.C = self.C_agg
        model_agg.fit( p, test_y, sample_weight=sample_weight, param_search=param_search, calibrate=self._calibrate )
        self._model_agg = (model_agg,quantiles)

    def predict(self, bags, y=None):
        """Predict bag label."""

        bags = [np.asmatrix(bag) for bag in bags]
        inst_preds = super(SIL, self).predict(np.vstack(bags))
        try:
            p = _inst_to_bag_preds(inst_preds, bags, self.predict_type, self.T, self._model_agg)
        except AttributeError:
            p = _inst_to_bag_preds(inst_preds, bags, self.predict_type, self.T)

        return p

    def score(self, bags, y=None, sample_weight=None):
        """Measure classifier performance; needed by sklearn grid search."""

        if type(y[0]) is tuple:
            y = [ yi[0] for yi in y ]

        p = self.predict( bags )
        d = np.argmax(p,axis=1)
        if self.metric == 'roc_auc':
            try:
                a = sklearn.metrics.roc_auc_score( y, p, sample_weight=sample_weight )
            except ValueError:
                a = 0
        elif self.metric == 'acc' or self.metric == 'accuracy':
            a = sklearn.metrics.accuracy_score( y, d, sample_weight=sample_weight )
        elif self.metric == 'kappa':
            a = sklearn.metrics.cohen_kappa_score( np.array(y), (d/2)+0.5, sample_weight=sample_weight )
        elif self.metric == 'balanced_accuracy':
            a = balanced_accuracy( y, d, sample_weight=sample_weight )
        
        return a

    def predict_instances(self, bags):
        """Predict class of instances."""

        return [super(SIL, self).predict(bag) for bag in bags]

    def param_search(self, bags, y, instances, classes, quick=True, C=1.0, gamma=1.0, bag_inst_idx=None, sample_weight=None, inst_search=False):
        """Search for best hyperparameters."""

        if bag_inst_idx is None:
            bag_inst_idx = [ [i]*len(b) for i,b in enumerate(bags) ]

        if C is None:
            # figure out an inital set of hyperparameters using the mean of all instances from each bag
            td = np.array([ t.mean(axis=0) for t in bags ])
            tl = np.array(y)

            # compute mean and std dev
            mu = td.mean(axis=0)
            sigma = td.std(axis=0) + 1e-3
            td = ( td - mu ) / sigma

            model = LinearClassifier( classifier=self.classifier, kernel=self.kernel, n_jobs=self.n_jobs )
            C,gamma = model.param_search( td, tl )

        acc = {}
        bestacc = 0
        bestg = None
        bestC = None
        while True:

            # start with values given and search in neighborhood; search will continue if best value falls on edge of neighborhood
            Cvals = [ float(2**e)*C for e in range(-2,3) ]
            if self.kernel == 'rbf':
                gvals = [ float(2**e)*gamma for e in range(-2,3) ]
            else:
                gvals = [1.0]

            # get instance indices for each bag
            idx = []
            i = 0
            for yi,inst in zip(y,bags):
                idx.append( np.arange(i,i+len(inst)) )
                i += len(inst)

            if self.kernel == 'rbf':
                Cvals2 = [ C for C in Cvals ]
            else:
                Cvals2 = [ C for C in Cvals if (C,1.0) not in acc.keys() ]

            folds = 5

            # grid search
            if inst_search:
                # find best instance-level classifier
                model = LinearClassifier( classifier=self.classifier, kernel=self.kernel, p=self.p, n_jobs=self.n_jobs )
                bestC,bestg = model.param_search( instances, classes, quick, C=C, gamma=gamma, sample_weight=sample_weight )
                bestacc = 0
            else:
                # find best result at bag level
                skf = sklearn.model_selection.StratifiedKFold( n_splits=folds, shuffle=True )
                labels = [ (y[i],np.array([classes[j] for j in idx[i]])) for i in range(len(y)) ]
                est = SIL( classifier=self.classifier, kernel=self.kernel, predict_type=self.predict_type, class_weight=self.class_weight, p=self.p, subset=self.subset, quantiles=self.quantiles, metric=self.metric )
                gridcv = sklearn.model_selection.GridSearchCV( est, [{'C':Cvals2,'gamma':gvals}], cv=skf, n_jobs=self.n_jobs, refit=False )
                gridcv.fit( bags, y, sample_weight=sample_weight, calibrate=self._calibrate )
                for mean_score,params in zip(gridcv.cv_results_['mean_test_score'],gridcv.cv_results_['params']):
                    acc[params['C'],params['gamma']] = mean_score
                if gridcv.best_score_ > bestacc:
                    bestC = gridcv.best_params_['C']
                    bestg = gridcv.best_params_['gamma']
                    bestacc = gridcv.best_score_

            if bestC == Cvals[0] or bestC == Cvals[-1] or ( self.kernel == 'rbf' and ( bestg == gvals[0] or bestg == gvals[-1] ) ):
                C = bestC
                gamma = bestg
            else:
                break

        self._model = None
        self._model_agg = None

        return bestC,bestg

def _inst_to_bag_preds(inst_preds, bags, predict_type='median', T=None, model_agg=None):
    """Predict bag class from instance predictions."""

    if 'quantile' in predict_type:
        if inst_preds.shape[1] == 2:
            r = np.array([model_agg[0].predict( np.percentile( inst_preds[slice(*bidx),1], model_agg[1] ).reshape(1,-1) )
                          for bidx in slices(map(len, bags))])
        else:
            r = np.array([model_agg[0].predict( np.array(np.hstack([ np.percentile( inst_preds[slice(*bidx),c], model_agg[1] ) for c in
                                                                     range(inst_preds.shape[1]) ]).reshape((1,-1)) ) ).flatten() for bidx in slices(map(len, bags))])
        return r.squeeze()
    elif predict_type == 'median':
        return np.array([np.median(inst_preds[slice(*bidx)],axis=0)
                         for bidx in slices(map(len, bags))])
    elif predict_type == 'max':
        return np.array([np.max(inst_preds[slice(*bidx)],axis=0)
                         for bidx in slices(map(len, bags))])
    else:
        print('Error: aggregation method %s no supported'%predict_type)
        sys.exit(1)

