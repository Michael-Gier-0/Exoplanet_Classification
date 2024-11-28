""" Michael Gier
    ITP-449
    Final Project
    This program takes a dataset of information about exoplanet candidates collected by the Kepler space telescope,
    wrangles the data, creates a seperate dataset that has undergone dimensionality reduction through PCA,
    uses cross validation to determine the optimal classification model to use, uses a GridSearch cross validation
    method to determine the optimal parameters for the classification model, trains the data using the optimal model and parameters,
    and reports how well the model classified the exoplanet candidates. 
"""
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay, classification_report

def main():

##### Step 1: Wrangle the data #####

    # collect data
    file_path = 'cumulative_2023.11.07_13.44.30.csv'
    df_exoplanets = pd.read_csv(file_path, skiprows=41)

    # drop duplicates
    df_exoplanets = df_exoplanets.drop_duplicates()

    # select attributes
    df_exoplanets = df_exoplanets[['koi_pdisposition', 'koi_period', 'koi_eccen', 'koi_duration',
                                'koi_prad', 'koi_sma', 'koi_incl', 'koi_teq', 'koi_dor', 'koi_steff',
                                'koi_srad', 'koi_smass']]
    
    # drop rows with null values
    df_exoplanets = df_exoplanets.dropna()

    # separate data and target
    X = df_exoplanets.drop(columns='koi_pdisposition')
    y = df_exoplanets['koi_pdisposition']
    (dim1, dim2) = X.shape

    # standardize data
    tfr_standard = StandardScaler()
    X_t = tfr_standard.fit_transform(X)
    X = pd.DataFrame(X_t, columns=X.columns, index=X.index)

    # determine optimal number of principle components
    exp_variance_list = []
    for i in range(1, dim2 + 1):
        components = i
        pca = PCA(n_components=components)
        pca = pca.fit(X)
        total_exp_variance = float(sum(pca.explained_variance_ratio_))
        exp_variance_list.append(total_exp_variance)

    opt_var_ratio = 0
    j = 0
    while float(opt_var_ratio) < 0.9:
        opt_var_ratio = float(exp_variance_list[j])
        j += 1
    opt_comp = j

    # create PCA dataset
    opt_pca = PCA(n_components=opt_comp)
    X_pca = pd.DataFrame(opt_pca.fit_transform(X), index=X.index)

    # partition into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=18, stratify=y)
    X_pca_train, X_pca_test, y_pca_train, y_pca_test = train_test_split(X_pca, y, random_state=18, stratify=y)

    # calculate maximum n_neighbors value
    max_n_neighbors = int(1.5 * (dim1 ** 0.5))

##### Step 2: Cross Validate #####

    # initialize the Pipeline
    pipe = Pipeline([('estimator', None)]) # placeholder for estimators

    # create list of estimators and hyperparameters
    estimator_list = [
    {'estimator': [LogisticRegression()]},

    {'estimator': [KNeighborsClassifier()],
    'estimator__n_neighbors': range(1, max_n_neighbors+1)},

    {'estimator': [DecisionTreeClassifier()],
    'estimator__criterion': ['entropy', 'gini'],
    'estimator__max_depth': range(3, 16),
    'estimator__min_samples_leaf': range(1, 11)},

    {'estimator': [SVC()],
    'estimator__kernel': ['rbf'],
    'estimator__C': (0.1, 1, 10, 100),
    'estimator__gamma': (0.1, 1, 10)}
    ]

    # find n_iter for RandomizedSearchCV, so that 10% of space is searched 
    hyperparam_space_dim = int(1 +
        len(estimator_list[1]['estimator__n_neighbors']) + 
        len(estimator_list[2]['estimator__criterion']) * len(estimator_list[2]['estimator__max_depth']) * len(estimator_list[2]['estimator__min_samples_leaf']) +
        len(estimator_list[3]['estimator__kernel']) * len(estimator_list[3]['estimator__C']) * len(estimator_list[3]['estimator__gamma']))
    max_iter = int(math.ceil(hyperparam_space_dim * 0.1))

    # use Randomized Search to select estimator
    model_list = []
    for i in range(2):
        rscv = RandomizedSearchCV(
            pipe,
            param_distributions=estimator_list,
            scoring='accuracy',
            n_iter=max_iter,
            random_state=18)

        # fit RandomizedSearchCV
        if i == 0:
            rscv.fit(X_train, y_train)
        else:
            rscv.fit(X_pca_train, y_pca_train)

        # store best model
        model_list.append(rscv.best_estimator_[0])

    # train models
    model_0 = model_list[0]
    model_1 = model_list[1]

    model_0.fit(X_train, y_train)
    model_1.fit(X_pca_train, y_pca_train)

    # use models to predict
    prediction_0 = model_0.predict(X_test)
    prediction_1 = model_1.predict(X_pca_test)

    # compare scores
    accuracy_score_0 = accuracy_score(y_test, prediction_0)
    accuracy_score_1 = accuracy_score(y_pca_test, prediction_1)
    
    # select better model
    if accuracy_score_0 <= accuracy_score_1:
        model = model_1
    else:
        model = model_0

    # initialize Grid Search to optimize parameters
    if 'LogisticRegression' in str(type(model)):
        parameter_grid = {}

    elif 'KNeighborsClassifier' in str(type(model)):
        parameter_grid = {'n_neighbors': range(1, max_n_neighbors+1)}

    elif 'DecisionTreeClassifier' in str(type(model)):
        parameter_grid = {'criterion': ['entropy', 'gini'],
            'max_depth': range(3, 16),
            'min_samples_leaf': range(1, 11)}

    elif 'SVC' in str(type(model)):
        parameter_grid = {'kernel': ['rbf'],
            'C': (0.1, 1, 10, 100),
            'gamma': (0.1, 1, 10)}

    gscv = GridSearchCV(estimator=model,
        param_grid=parameter_grid,
        scoring='accuracy')

    if model == model_0:
        gscv.fit(X_train, y_train)
        opt_model = gscv.best_estimator_
    else:
        gscv.fit(X_pca_train, y_pca_train)
        opt_model = gscv.best_estimator_

##### Step 3: Train Optimized Model #####

    # train model and create prediction
    if model == model_0:
        opt_model.fit(X_train, y_train)
        prediction = opt_model.predict(X_test)
        cm = confusion_matrix(y_test, prediction)
        class_rep = classification_report(y_test, prediction)
    else:
        opt_model.fit(X_pca_train, y_pca_train)
        prediction = opt_model.predict(X_pca_test)
        cm = confusion_matrix(y_pca_test, prediction)
        class_rep = classification_report(y_pca_test, prediction)

    # display confusion matrix
    cm_disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Candidate', 'False Positive'])
    fig, axes = plt.subplots()
    cm_disp.plot(ax=axes)
    axes.set(title='Confusion Matrix for Exoplanet Candidate Classification Model')

    # save figure
    fig.tight_layout()
    fig.savefig('Exoplanet_confusion_matrix.png')

    # display classification report
    print(class_rep)


if __name__ == '__main__':
    main()

