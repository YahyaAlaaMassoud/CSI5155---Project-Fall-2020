import numpy as np
import matplotlib.pyplot as plt
import random
import copy
import time

from itertools import product
from sklearn.tree import DecisionTreeClassifier # Import Decision Tree Classifier
from sklearn.naive_bayes import GaussianNB # Import Naive Bayes Classifier
from sklearn.svm import SVC, LinearSVC # Import SVM Classifier
from sklearn.neighbors import KNeighborsClassifier # Import KNN Classifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn.metrics import confusion_matrix, multilabel_confusion_matrix, accuracy_score, balanced_accuracy_score, precision_recall_fscore_support, roc_auc_score #Import scikit-learn metrics module for accuracy calculation
from sklearn import tree
from pprint import pprint
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.under_sampling import ClusterCentroids
from imblearn.combine import SMOTEENN, SMOTETomek
from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold
from prettytable import PrettyTable
from scipy import stats
from matplotlib.pyplot import pie, axis, show
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer, SimpleImputer
from sklearn.feature_selection import VarianceThreshold, SelectKBest, mutual_info_classif, SelectFromModel, RFE
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier
from sklearn.model_selection import cross_validate
from sklearn.metrics import recall_score

random.seed(0)

def plot_conf_mat(mat, title):
    # Viewing the confusion matrix
    plt.imshow(mat, interpolation='nearest')
    plt.title(title + ' Confusion Matrix')
    plt.colorbar()
    plt.show()

def train_model(model_name, X, y, folds, data_classes):
    train_time   = []
    conf_mat     = np.zeros((len(data_classes),len(data_classes)))
    recall       = []
    percision    = []
    f1_score     = []
    support      = []
    acc          = []
    balanced_acc = []
    neg_recall   = []

    for train, test in folds:
        if model_name == 'XGBoost':
            clf = GradientBoostingClassifier(random_state=0)
        elif model_name == 'AdaBoost':
            clf = AdaBoostClassifier(random_state=0)
        elif model_name == 'XTrees':
            clf = ExtraTreesClassifier()
        elif model_name == 'DT':
            clf = DecisionTreeClassifier(max_depth=5)
        elif model_name == 'KNN':
            clf = KNeighborsClassifier(n_neighbors=7)
        else:
            return None
        
        X_train = X[train]
        y_train = y[train]
        X_test  = X[test]
        y_test  = y[test]

        # Training
        tic = time.time()
        clf.fit(X_train, y_train)
        toc = time.time()
        train_time.append(np.round(toc - tic, 3))

        y_pred = clf.predict(X_test)

        # Compute confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        conf_mat = conf_mat + cm
        
        FP = cm.sum(axis=0) - np.diag(cm)  
        FN = cm.sum(axis=1) - np.diag(cm)
        TP = np.diag(cm)
        TN = cm.sum() - (FP + FN + TP)
        
        # Specificity or true negative rate
        TNR = TN/(TN+FP) 
        neg_recall.append(TNR)

        p, r, f1, sup = precision_recall_fscore_support(y_test, y_pred)
        percision.append(p)
        recall.append(r)
        f1_score.append(f1)
        support.append(sup)

        acc.append(accuracy_score(y_test, y_pred))
        balanced_acc.append(balanced_accuracy_score(y_test, y_pred))
        
    tables = []
    t = PrettyTable([model_name, 'Avg. Accross 10 Folds'])
    t.add_row(['Training Time', np.mean(train_time)])
    t.add_row(['Accuracy', np.mean(acc)])
#     t.add_row(['Balanced Accuracy', np.mean(balanced_acc)])
    tables.append(t)
    
    percision = np.array(percision)
    recall    = np.array(recall)
    f1_score  = np.array(f1_score)
    support   = np.array(support)

    t = PrettyTable(['Metric / Class', 'Class - No', 'Class - > 30 days', 'Class - < 30 days'])
    t.add_row(['Percision'] + np.mean(percision, axis=0).tolist())
    t.add_row(['Recall'] + np.mean(recall, axis=0).tolist())
    t.add_row(['Specificity'] + np.mean(neg_recall, axis=0).tolist())
    t.add_row(['F1-score'] + np.mean(f1_score, axis=0).tolist())
#     t.add_row(['Support'] + np.mean(support, axis=0).tolist())
    tables.append(t)
    
    conf_mat = conf_mat.T # since rows and  cols are interchanged

    t = PrettyTable(['', 'No', '> 30 days', '< 30 days'])
    t.add_row(['No'] + conf_mat[0].tolist())
    t.add_row(['> 30 days'] + conf_mat[1].tolist())
    t.add_row(['< 30 days'] + conf_mat[2].tolist())
    tables.append(t)
    
    avg_acc = np.trace(conf_mat) / sum(data_classes)
    conf_mat_norm = conf_mat / data_classes # Normalizing the confusion matrix
    tables.append(conf_mat_norm)
    
    metrics = {
        'clf': clf,
        'percision': percision,
        'recall': recall,
        'f1_score': f1_score,
        'support': support,
        'conf_mat': conf_mat,
        'acc': acc,
        'balanced_acc': balanced_acc,
        'train_time': train_time
    }
    
    return tables, metrics