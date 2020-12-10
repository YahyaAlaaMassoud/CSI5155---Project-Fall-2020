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
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
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
from sklearn.metrics import recall_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
from scipy import interp

random.seed(0)

def plot_pie_chart(labels, y, n_features):
    # Pie chart, where the slices will be ordered and plotted counter-clockwise:
    labels = ['No', '>30', '<30']
    sizes = []
    for i in range(len(labels)):
        sizes.append(len(y[y==i]))
    explode = []
    for i in range(len(labels)):
        explode.append(sizes[i] / sum(sizes) / 10)

    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.title('Combined Sampling performed after Selection of {} Features'.format(n_features))
    plt.show()

def plot_roc_macro(all_metrics, all_clf, cls2clr):
    lw = 2
    for i, (metrics, clf) in enumerate(zip(all_metrics, all_clf)):
        plt.plot(metrics['fpr']['macro'], metrics['tpr']['macro'], color=cls2clr[i],
                 lw=lw, label='ROC curve - Classifier: {} (area = {})'.format(clf, np.round(metrics['roc_auc']['macro'], 2)))
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic OvR - Macro Average')
    plt.legend(loc="lower right")
    plt.show()
    
def plot_roc_micro(all_metrics, all_clf, cls2clr):
    lw = 2
    for i, (metrics, clf) in enumerate(zip(all_metrics, all_clf)):
        plt.plot(metrics['fpr']['micro'], metrics['tpr']['micro'], color=cls2clr[i],
                 lw=lw, label='ROC curve - Classifier: {} (area = {})'.format(clf, np.round(metrics['roc_auc']['micro'], 2)))
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic OvR - Micro Average')
    plt.legend(loc="lower right")
    plt.show()

def plot_roc(metrics, data_classes, cls2clr, title):
    lw = 2
    for i in range(len(data_classes)):
        plt.plot(metrics['fpr'][i], metrics['tpr'][i], color=cls2clr[i],
                 lw=lw, label='ROC curve - Class: {} (area = {})'.format(i, np.round(metrics['roc_auc'][i], 2)))
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic OvR ' + title)
    plt.legend(loc="lower right")
    plt.show()

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
    
    # ROC COMPUTATION
    # shuffle and split training and test sets
    y = label_binarize(y, classes=list(range(len(data_classes))))
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.1, random_state=0, stratify=y)
    
    if model_name == 'XGBoost':
        clf = OneVsRestClassifier(GradientBoostingClassifier(random_state=0))
    elif model_name == 'AdaBoost':
        clf = OneVsRestClassifier(AdaBoostClassifier(random_state=0))
    elif model_name == 'XTrees':
        clf = OneVsRestClassifier(ExtraTreesClassifier())
    elif model_name == 'DT':
        clf = OneVsRestClassifier(DecisionTreeClassifier(max_depth=5))
    elif model_name == 'KNN':
        clf = OneVsRestClassifier(KNeighborsClassifier(n_neighbors=7))
    else:
        return None
    
    y_score = clf.fit(X_train, y_train).predict_proba(X_test)
    n_classes = len(data_classes)
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
        
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(n_classes):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

    # Finally average it and compute AUC
    mean_tpr /= n_classes

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

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
        'specificity': neg_recall,
        'support': support,
        'conf_mat': conf_mat,
        'acc': acc,
        'balanced_acc': balanced_acc,
        'train_time': train_time,
        'roc_auc': roc_auc,
        'fpr': fpr,
        'tpr': tpr,
    }
    
    return tables, metrics