import numpy as np
import matplotlib.pyplot as plt
import random
import copy
import time
import pandas as pd

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
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from pypcc import ParticleCompetitionAndCooperation

random.seed(0)

def plot_pie_chart(labels, y, n_features, task='sl'):
    # Pie chart, where the slices will be ordered and plotted counter-clockwise:
    sizes = []
    if task is 'sl':
        for i in range(len(labels)):
            sizes.append(len(y[y==i]))
    elif task is 'ssl':
        for i in range(-1, len(labels) - 1):
            sizes.append(len(y[y==i]))
    explode = []
    for i in range(len(labels)):
        explode.append(sizes[i] / sum(sizes) / 10)

    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    if task is 'sl':
        plt.title('Combined Sampling performed after Selection of {} Features'.format(n_features))
    elif task is 'ssl':
        plt.title('Perecentage of data after unlabelling {}%'.format(n_features))
    plt.show()

def plot_roc_avg(all_metrics, all_clf, cls2clr, macro=False, micro=True):
    m = 'micro' if micro else 'macro'
    lw = 2
    for i, (metrics, clf) in enumerate(zip(all_metrics, all_clf)):
        plt.plot(metrics['fpr'][m], metrics['tpr'][m], color=cls2clr[i],
                 lw=lw, label='ROC curve - Classifier: {} (area = {})'.format(clf, np.round(metrics['roc_auc'][m], 2)))
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic OvR - {} Average'.format(m))
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
    
def stats_test(all_scores, all_clf):
    all_pairs_idx = set()
    for i in range(5):
        for j in range(i + 1, 5):
            all_pairs_idx.add((i, j))
    K = 10
    paired_ttest = {}
    for pair in all_pairs_idx:
        clf1, clf2 = pair
        score_diff = []
        for i in range(K):
            score_diff.append(all_scores[clf1][i] - all_scores[clf2][i])
        t, p = stats.ttest_rel(all_scores[clf1], all_scores[clf2])
        paired_ttest[pair] = {
            'score_diff': score_diff,
            't_stat': np.round(t, 2), 
            'p_value': np.round(p, 8), 
        }
    t = PrettyTable(['Classifier-1 vs Classifier-2', 'avg_diff', 'stdev_diff', 'pvalue'])#, 'tvalue'])
    for pair in all_pairs_idx:
        clf1, clf2 = pair
        t.add_row([all_clf[clf1] + ' vs ' + all_clf[clf2], 
                   "{:0.2f}".format(np.mean(paired_ttest[pair]['score_diff'])), 
                   "{:0.2f}".format(np.std(paired_ttest[pair]['score_diff'])), 
    #                paired_ttest[pair]['t_stat'],
                   np.sum(paired_ttest[pair]['p_value'])])
    return t

class Trainer:
    
    def __init__(self, X, y, folds, data_classes, classes):
        self.X = X
        self.y = y
        self.folds = folds
        self.data_classes = data_classes
        self.classes = classes
    
    def train_model(self, model_name):
        ssl = False
        if model_name in ['LabelProp', 'LabelSpread', 'PYCC']:
            ssl = True
            
        train_time   = []
        conf_mat     = np.zeros((len(self.data_classes), len(self.data_classes)))
#         print('conf_mat.shape', conf_mat.shape)
        recall       = []
        percision    = []
        f1_score     = []
        support      = []
        acc          = []
        balanced_acc = []
        neg_recall   = []

        for train, test in self.folds:
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
            elif model_name == 'LabelProp':
                clf = LabelPropagation(kernel='knn')
            elif model_name == 'LabelSpread':
                clf = LabelSpreading(kernel='knn')
            elif model_name == 'PYCC':
                clf = ParticleCompetitionAndCooperation(n_neighbors=1, pgrd=0.6, delta_v=0.35, max_iter=1000)
            else:
                return None

            X_train = self.X[train]
            y_train = self.y[train]
            X_test  = self.X[test]
            y_test  = self.y[test]
            
            if ssl:
                unlabelled_idx = np.where(y_test==-1)
#                 print('y_test.shape, X_test.shape', y_test.shape, X_test.shape)
                X_test = np.delete(X_test, unlabelled_idx, axis=0)
                y_test = np.delete(y_test, unlabelled_idx)
#                 print('y_test.shape, X_test.shape', y_test.shape, X_test.shape)

            # Training
            tic = time.time()
            clf.fit(X_train, y_train)
            toc = time.time()
            train_time.append(np.round(toc - tic, 3))
            
            y_pred = clf.predict(X_test)
#             print('np.unique(y_pred), np.unique(y_test)', np.unique(y_pred), np.unique(y_test))

            # Compute confusion matrix
            cm = confusion_matrix(y_test, y_pred)
#             print('cm.shape', cm.shape)
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
        b = np.zeros((self.y.size, self.y.max() + 1))
        b[np.arange(self.y.size), self.y] = 1
        y = b
        X_train, X_test, y_train, y_test = train_test_split(self.X, y, test_size=.1, random_state=0, stratify=y)

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
        elif model_name == 'LabelProp':
            clf = OneVsRestClassifier(LabelPropagation(kernel='knn'))
        elif model_name == 'LabelSpread':
            clf = OneVsRestClassifier(LabelSpreading(kernel='knn'))
        elif model_name == 'PYCC':
            clf = OneVsRestClassifier(ParticleCompetitionAndCooperation(n_neighbors=1, pgrd=0.6, delta_v=0.35, max_iter=1000))
        else:
            return None

        if model_name is not 'PYCC':
            y_score = clf.fit(X_train, y_train).predict_proba(X_test)
        else:
            y_score = clf.fit(X_train, y_train).predict(X_test)
#         print('y_test.shape, y_score.shape', y_test.shape, y_score.shape)
        n_classes = len(self.data_classes)
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

        t = PrettyTable(['Metric / Class'] + self.classes)
        t.add_row(['Percision'] + np.mean(percision, axis=0).tolist())
        t.add_row(['Recall'] + np.mean(recall, axis=0).tolist())
        t.add_row(['Specificity'] + np.mean(neg_recall, axis=0).tolist())
        t.add_row(['F1-score'] + np.mean(f1_score, axis=0).tolist())
    #     t.add_row(['Support'] + np.mean(support, axis=0).tolist())
        tables.append(t)

        conf_mat = conf_mat.T # since rows and  cols are interchanged

        t = PrettyTable([''] + self.classes)
        for i, cls in enumerate(self.classes):
            t.add_row([cls] + conf_mat[i].tolist())
        tables.append(t)

        avg_acc = np.trace(conf_mat) / sum(self.data_classes)
        conf_mat_norm = conf_mat / (self.data_classes) # Normalizing the confusion matrix
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
            'roc_auc': roc_auc or [],
            'fpr': fpr or [],
            'tpr': tpr or [],
        }

        return tables, metrics
    
def read_and_process_data(data_path):
    data = pd.read_csv(data_path, sep=',')
    print('---Retreived data from {}'.format(data_path))
    
    print('Number of records:', data.shape[0], 'Number of features:', data.shape[1])
    
    t = PrettyTable(['Feature', 'Number of Unique Elements'])
    # list the unique elements in columns
    for col in data.columns:
        t.add_row([col, len(data[col].unique())])
    print(t)
    
    data.replace(to_replace="Unknown/Invalid", value=np.nan, inplace=True)
    data.replace(to_replace="?", value=np.nan, inplace=True)
    print('---Replacing ? -> np.nan')
    data = data[data['diag_1'].notna()]
    data = data[data['diag_2'].notna()]
    data = data[data['diag_3'].notna()]
    data = data[data['race'].notna()]
    print('---Removing rows where diag_1, diag_2, diag_3, race are missing')

    print('Percentage (%) of Missing values in each feature')
    print('--------------------------------------------')
    print(np.round(data.isnull().sum() / len(data) * 100, 1))
    print('--------------------------------------------')
    print('--------------------------------------------')
    
    del data['weight'], data['payer_code'], data['encounter_id'], data['patient_nbr']
    print('---Removing columns weight, payer_code, encounter_id, patient_nbr')
    
#     data['readmitted'].value_counts().plot.pie()
#     data['gender'].value_counts().plot.pie()
    
    col_names = data.columns
    
    data.diag_1 = data.diag_1.astype(str)
    data.diag_2 = data.diag_2.astype(str)
    data.diag_3 = data.diag_3.astype(str)
    diag_dict = {}
    # Circulatory
    diag_dict[785] = 'Circulatory'
    for i in range(390, 459):
        diag_dict[i] = 'Circulatory'
    # Respiratory
    diag_dict[786] = 'Respiratory'
    for i in range(460, 519):
        diag_dict[i] = 'Respiratory'
    # Digestive
    diag_dict[787] = 'Digestive'
    for i in range(520, 579):
        diag_dict[i] = 'Digestive'
    # Injury
    for i in range(800, 999):
        diag_dict[i] = 'Injury'
    # Musculoskeletal
    for i in range(710, 739):
        diag_dict[i] = 'Musculoskeletal'
    # Genitourinary
    for i in range(580, 629):
        diag_dict[i] = 'Genitourinary'
    # Neoplasms
    diag_dict[780] = 'Neoplasms'
    diag_dict[781] = 'Neoplasms'
    diag_dict[782] = 'Neoplasms'
    diag_dict[784] = 'Neoplasms'
    for i in range(140, 239):
        diag_dict[i] = 'Neoplasms'
    for i in range(790, 799):
        diag_dict[i] = 'Neoplasms'
    for i in range(240, 279):
        if i is 250:
            continue
        diag_dict[i] = 'Neoplasms'
    for i in range(680, 709):
        diag_dict[i] = 'Neoplasms'
    # Diabetes
    for i in data.diag_1:
        if i.startswith("250"):
            diag_dict[i] = 'Diabetes'
    for i in data.diag_1:
        if i not in diag_dict:
            diag_dict[i] = 'Other'

    final_dict = {}
    for k, v in diag_dict.items():
        final_dict[str(k)] = v

    final_dict = pd.Series(final_dict)
    data.diag_1 = data.diag_1.map(final_dict)
    data.diag_2 = data.diag_2.map(final_dict)
    data.diag_3 = data.diag_3.map(final_dict)
    # print(len(data.diag_1.unique()))
    # print(len(data.diag_2.unique()))
    # print(len(data.diag_3.unique()))
    
    # label encoding for only columns with categorical data
    data.replace(to_replace=np.nan, value='?', inplace=True)
    for col in data.columns:
        if col not in ['time_in_hospital', 
                       'num_lab_procedures', 
                       'num_procedures', 
                       'num_medications', 
                       'number_outpatient', 
                       'number_emergency', 
                       'number_inpatient', 
                       'number_diagnoses']:
            unique_elem = sorted(data[col].unique())
            if '?' in unique_elem:
                unique_elem = unique_elem[1:]
            data[col].replace(unique_elem, list(range(len(unique_elem))), inplace=True)
    data.replace(to_replace='?', value=np.nan, inplace=True)
    print('---Transform nominal data to categorical')
    
    imputed_data = IterativeImputer(max_iter=100, random_state=0, initial_strategy='most_frequent').fit_transform(data.to_numpy())
    # imputed_data = SimpleImputer(strategy='most_frequent').fit_transform(data.to_numpy())
    data = pd.DataFrame.from_records(imputed_data)
    data.columns = col_names
    print('---Imputing missing values')
    
    ids = data.columns[:-1]
    data[ids] = (data[ids] - data[ids].mean()) / (data[ids].std() + 1e-4)
    print('---Standarizing data')
    
    return data