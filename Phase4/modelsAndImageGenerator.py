from sklearn.model_selection import cross_validate,GridSearchCV
from sklearn.metrics import log_loss, accuracy_score, balanced_accuracy_score, make_scorer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from scipy.io.arff import loadarff
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from joblib import dump, load
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
import csv
from sklearn.metrics import precision_recall_fscore_support, make_scorer,precision_score,recall_score
from os import mkdir
from os.path import exists


import numpy as np


def make_dir(directory):
    if not exists(directory):
        mkdir(directory);

def save_img(name):
    plt.savefig(pdf_folder + name + ".pdf", format = 'pdf', bbox_inches='tight', dpi=1200)                              
    plt.savefig(eps_folder + name + ".eps", format = 'eps', dpi=1200)
    plt.savefig(jpg_folder + name + ".jpg", format = 'jpg', dpi=1200)
    plt.savefig(tiff_folder + name + ".tiff", format = 'tiff', dpi = 1200)


def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5), scoring = 'accuracy'):
    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    axes : array of 3 axes, optional (default=None)
        Axes to use for plotting the curves.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes.set_title(title)
    if ylim is not None:
        axes.set_ylim(*ylim)
    axes.set_xlabel("Training examples")
    axes.set_ylabel("balanced accuracy")


    if not loading:
        train_sizes, train_scores, test_scores, = \
            learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                           train_sizes=train_sizes, scoring = scoring, verbose = 3)
        
        dump([train_sizes,train_scores, test_scores], model_folder + "learning_curve_data_" + modelName + ".joblib")
    else:
        train_sizes, train_scores, test_scores = load(model_folder + "learning_curve_data_" + modelName + ".joblib");

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # Plot learning curve
    axes.grid()
    axes.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes.plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes.legend(loc="best")

    return plt
       




selected_models = [1,2,3,4,5,6,7,8]
selection2 = 0;
train_sizes= [0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 0.75, 1.]
grids_cv = 5;
learn_cv = 10;
loading = True;
do_learning_curves = True;


folder = "Latest_experiments/";


font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 17}

mpl.rc('font', **font)
mpl.rcParams['lines.linewidth'] = 2

#scoring = 'accuracy'
#scoring = 'f1_weighted'
scoring = 'balanced_accuracy'

#scoring = 'brier_score_loss'

img_folder = folder + "images/"
model_folder = folder+ "models/"
metrics_folder = folder + "metrics/"
pdf_folder = img_folder + "pdf/";
eps_folder = img_folder + "eps/";
jpg_folder = img_folder + "jpg/";
tiff_folder = img_folder + "tiff/";
make_dir(folder);
make_dir(img_folder);
make_dir(model_folder);
make_dir(metrics_folder);
make_dir(pdf_folder);
make_dir(eps_folder);
make_dir(jpg_folder);
make_dir(tiff_folder);



###MODIFIED BY PAOLO
clf=""
print("Loading dataset..")

datasetName = "datasets/Aggregate-dataset.arff"

dataset = loadarff(open(datasetName,'r'))

X = np.array(dataset[0][['tensor_1', 'tensor_2', 'tensor_3', 'tensor_4', 'tensor_5','tensor_6','tensor_7','tuner','precision','architecture']],dtype=[('tensor_1', float),('tensor_2', float),('tensor_3', float),		('tensor_4', float),('tensor_5', float),('tensor_6', float),('tensor_7', float),('tuner',float),('precision',float),('architecture',float)])

X = X.view((float, len(X.dtype.names)))
y = np.array(dataset[0]['class'])


print(len(X))
metrics_file = open('CONV_metrics.csv', mode='w');
metrics_writer = csv.writer(metrics_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
metrics_writer.writerow(['Model','accuracy','acc_err','balanced','bacc_err','conv_prec','cp_err','conv_rec','cr_err','d_conv_prec','dcp_err','d_conv_rec','dcr_err','w_conv_prec','wcp_err','w_conv_rec','wcr_err'])

    

cv = ShuffleSplit(n_splits=learn_cv, test_size=0.2, random_state=0)

for selection1 in selected_models:
    
    
    if(selection1==1):
        clf = RandomForestClassifier(n_estimators=300, max_depth=8,class_weight = 'balanced')
        if selection2 == 0:
            Ns = [10,50,100,300];
            depths = [3,5,10,20];
            params = {'n_estimators':Ns,'max_depth' : depths};
            clf = GridSearchCV(clf, params, cv=grids_cv, scoring = scoring);
        modelName="RF"
        
    elif(selection1==2):
        clf = MLPClassifier(hidden_layer_sizes = (30,50,30))
        if selection2 == 0:
            parameter_space = {
                    'hidden_layer_sizes': [(50,50), (30,50,30), (100,)],
                    'activation': ['tanh', 'relu'],
                    'solver': ['sgd', 'adam'],
                    'alpha': [0.0001, 0.05],
                    'learning_rate': ['constant','adaptive']
                    }
            clf = GridSearchCV(clf, parameter_space, cv=grids_cv, scoring = scoring);
        modelName="MLP"
            
    elif(selection1==3):
        clf = DecisionTreeClassifier(class_weight = 'balanced')
        if selection2 == 0:
            depths = [5,10,25,50,100];
            splits = [2,3,5,10,50];
            crits = ['gini','entropy']
            params = {'max_depth' : depths, 'min_samples_split': splits, 'criterion' : crits};
            clf = GridSearchCV(clf, params, cv=grids_cv, scoring = scoring);
        modelName="DT"
            
    elif(selection1==4):
        clf = LogisticRegression(random_state=0, solver='newton-cg', C=0.5, multi_class='multinomial',max_iter=10000,class_weight = None)
        if selection2 == 0:
            Cs = [0.1,0.5,1.,2,100]
            class_weight_s = ['balanced',None]
            solvers = ['newton-cg','lbfgs']
            params = {'C' : Cs, 'class_weight': class_weight_s,'solver':solvers};
            clf = GridSearchCV(clf, params, cv=grids_cv, scoring = scoring);
        modelName="LoR"

    elif(selection1==5):
        clf = KNeighborsClassifier(n_neighbors = 3, algorithm = 'ball_tree');
        if selection2 == 0:
            NN = [2,3,5,10,30];
            ws = ['uniform','distance'];
            algos = ['ball_tree','kd_tree'];
            params = {'n_neighbors': NN, 'weights' : ws, 'algorithm': algos};
            clf = GridSearchCV(clf, params, cv=grids_cv, scoring = scoring);
        modelName="KNN"
            
    elif(selection1==6):
        	clf = GaussianNB(priors=None, var_smoothing=1e-8)
        	modelName="NBC"
            
    elif(selection1==7):
        clf = SVC(kernel='rbf', C = 8, gamma = 0.1, class_weight='balanced', random_state=0 , max_iter=100000)
        if selection2 == 0:
            Cs = [2,5,10,25];
            gammas = [0.06,0.1,0.15,0.5];
            params = {'C': Cs, 'gamma' : gammas};
            clf = GridSearchCV(clf, params, cv=grids_cv, scoring = scoring);
        modelName="SVM"
    
    elif(selection1==8):
        clf = GradientBoostingClassifier(n_estimators=100, learning_rate=0.5, max_depth=5, random_state=0)
        if selection2 == 0:
            ns = [50,100,200,300];
            learnings = [0.1,0.5,1];
            depths = [1,5,10,50];
            params = {'n_estimators': ns,'learning_rate': learnings, 'max_depth':depths}
            clf = GridSearchCV(clf, params, cv=grids_cv, scoring = scoring);
        modelName="GTB"
    
    print('*'*10)
    print(modelName);

#            clf.fit(X, target)
    
    fig, axes = plt.subplots();
    title = modelName + ": Learning Curve"
    
    
    def rec_0(y_true, y_pred): return recall_score(y_true, y_pred, average = None)[0]
    def rec_1(y_true, y_pred): return recall_score(y_true, y_pred, average = None)[1]
    def rec_2(y_true, y_pred): return recall_score(y_true, y_pred, average = None)[2]

    def prec_0(y_true, y_pred): return precision_score(y_true, y_pred, average = None)[0]
    def prec_1(y_true, y_pred): return precision_score(y_true, y_pred, average = None)[1]
    def prec_2(y_true, y_pred): return precision_score(y_true, y_pred, average = None)[2]

    scores = {
                'accuracy': make_scorer(accuracy_score),
                'balanced_accuracy': make_scorer(balanced_accuracy_score),

                'prec_0': make_scorer(prec_0),
                'rec_0': make_scorer(rec_0), 

                'prec_1': make_scorer(prec_1),
                'rec_1': make_scorer(rec_1),

                'prec_2': make_scorer(prec_2),
                'rec_2': make_scorer(rec_2),
                }
    
    if not loading:
        cval_results = cross_validate(clf,X,y, scoring = scores, cv = cv, verbose = 3, n_jobs = -1);
        del cval_results['fit_time']
        del cval_results['score_time']
        dump(cval_results, metrics_folder + "metrics_" + modelName + ".joblib")
    else:
        cval_results = load(metrics_folder + "metrics_" + modelName + ".joblib");
    
    row = [modelName,]
    print(modelName)
    for k in cval_results.keys():
        print("ROW!")
        row.append(np.mean(cval_results[k]))
        row.append(np.std(cval_results[k]))
    metrics_writer.writerow(row)
    metrics_writer.writerow("wooow")
    
    if do_learning_curves:
        
        plot_learning_curve(clf, title, X, y, axes=axes, train_sizes = train_sizes, ylim=(0.2, 1.01),
            cv=cv, n_jobs=-1, scoring = scoring)
    
        img_name = modelName + "_learning_curve";
        save_img(img_name);


metrics_file.close()
    
    
    
    
