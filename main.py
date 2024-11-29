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
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report


def main():
    """
    Main function to optimize and train the classification model
    """
    filepath = 'cumulative_2023.11.07_13.44.30.csv'

    # wrangle the data
    [X, y] = wrangle_data(filepath)

    # create PCA dataset
    num_components = optimal_comp(X, 0.90)
    X_pca = pca_dataset(X, num_components)

    # partition into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=18, stratify=y)
    X_pca_train, X_pca_test, y_pca_train, y_pca_test = train_test_split(X_pca, y, random_state=18, stratify=y)

    # calculate maximum n_neighbors value
    max_n_neighbors = int(1.5 * (X.shape[0] ** 0.5))

    # define hyperparameter bounds
    hyperparams = {'logistic_regression_parameters': {},
                   'knn_parameters':
                       {'n_neighbors': range(1, max_n_neighbors + 1)},
                   'decision_tree_parameters':
                       {'criterion': ['gini', 'entropy'],
                        'max_depth': range(3, 16),
                        'min_samples_leaf': range(1, 11)},
                   'svc_parameters':
                       {'kernel': ['rbf'],
                        'C': [0.1, 1, 10, 100],
                        'gamma': [0.01, 1, 10]}}

    # create estimator list
    estimator_list = hyperparam_space(hyperparams)

    # calculate maximum number of iterations
    max_iter = find_max_iter(estimator_list, 0.10)

    # use Randomized Search to select estimators for original and pca datasets
    [orig_model, orig_score] = select_best_model(X_train, y_train, estimator_list, max_iter)
    [pca_model, pca_score] = select_best_model(X_pca_train, y_pca_train, estimator_list, max_iter)

    # select best model
    if orig_score > pca_score:
        best_model = orig_model
        pca = False
    else:
        best_model = pca_model
        pca = True

    # select train/test partitions
    if pca is True:
        X_train, X_test, y_train, y_test = X_pca_train, X_pca_test, y_pca_train, y_pca_test
    else:
        pass

    # use Grid Search to obtain optimal parameters
    opt_model = select_best_parameters(best_model, hyperparams, X_train, y_train)

    # evaluate optimized model
    evaluate_model(opt_model, X_train, y_train, X_test, y_test)


def wrangle_data(file_path):
    """
    Function to load and wrangle the exoplanet data
    :param file_path: path to exoplanet data file, string
    :return X: exoplanet data, DataFrame
    :return y: exoplanet target, DataFrame
    """
    # collect data
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

    # standardize data
    tfr_standard = StandardScaler()
    X_t = tfr_standard.fit_transform(X)
    X = pd.DataFrame(X_t, columns=X.columns, index=X.index)
    return [X, y]


def optimal_comp(X, min_var_ratio):
    """
    Function to determine the minimum number of principal components to retain a specified minimum variance ratio
    :param X: exoplanet data, DataFrame
    :param min_var_ratio: minimum variance ratio required, float
    :return opt_comp: the minimum number of principal components to retain specified minimum variance ratio, int
    """
    # determine optimal number of principle components
    exp_variance_list = []
    for i in range(1, X.shape[1] + 1):
        pca = PCA(n_components=i).fit(X)
        total_exp_variance = float(sum(pca.explained_variance_ratio_))
        exp_variance_list.append(total_exp_variance)

    # find the fewest components with total explained variance above threshold
    opt_var_ratio = 0
    j = 0
    while float(opt_var_ratio) < min_var_ratio:
        opt_var_ratio = float(exp_variance_list[j])
        j += 1
    opt_comp = j
    return opt_comp


def pca_dataset(X, opt_comp):
    """
    Function to transform the dataset using PCA with the optimal number of principal components
    :param X: exoplanet data, DataFrame
    :param opt_comp: the optimal number of principal components, int
    :return X_pca: transformed exoplanet data, DataFrame
    """
    # create PCA dataset
    opt_pca = PCA(n_components=opt_comp)
    X_pca = pd.DataFrame(opt_pca.fit_transform(X), index=X.index)
    return X_pca


def hyperparam_space(hyperparams):
    """
    Function to define the estimator list
    :param hyperparams: dictionary of hyperparameter values, dictionary
    :return estimator_list: list of hyperparameter value dictionaries corresponding to each estimator type, list
    """
    estimator_list = [
        {'estimator': [LogisticRegression()]},

        {'estimator': [KNeighborsClassifier()],
         'estimator__n_neighbors': hyperparams['knn_parameters']['n_neighbors']},

        {'estimator': [DecisionTreeClassifier()],
         'estimator__criterion': hyperparams['decision_tree_parameters']['criterion'],
         'estimator__max_depth': hyperparams['decision_tree_parameters']['max_depth'],
         'estimator__min_samples_leaf': hyperparams['decision_tree_parameters']['min_samples_leaf']},

        {'estimator': [SVC()],
         'estimator__kernel': hyperparams['svc_parameters']['kernel'],
         'estimator__C': hyperparams['svc_parameters']['C'],
         'estimator__gamma': hyperparams['svc_parameters']['gamma']},
    ]
    return estimator_list


def find_max_iter(estimator_list, min_space_ratio):
    """
    Function to find the minimum number of iterates needed to cover a specified
    fraction of the total hyperparameter space
    :param estimator_list: list of hyperparameter value dictionaries corresponding to each estimator type, list
    :param min_space_ratio: specified ratio of hyperparameter space, float
    :return max_iter: number of iterations needed to reach specified ratio of hyperparameter space, int
    """
    # find n_iter for RandomizedSearchCV, so that 10% of space is searched
    hyperparam_space_dim = int(1 + len(estimator_list[1]['estimator__n_neighbors']) +
                               len(estimator_list[2]['estimator__criterion']) *
                               len(estimator_list[2]['estimator__max_depth']) *
                               len(estimator_list[2]['estimator__min_samples_leaf']) +
                               len(estimator_list[3]['estimator__kernel']) *
                               len(estimator_list[3]['estimator__C']) *
                               len(estimator_list[3]['estimator__gamma']))
    max_iter = int(math.ceil(hyperparam_space_dim * min_space_ratio))
    return max_iter


def select_best_model(X_train, y_train, estimator_list, max_iter):
    """
    Function to select a best estimator based on Randomized Search Cross Validation
    :param X_train: training data, DataFrame
    :param y_train: training target, DataFrame
    :param estimator_list: list of hyperparameter value dictionaries corresponding to each estimator type, list
    :param max_iter: number of iterations needed to reach specified ratio of hyperparameter space, int
    :return model: selected best estimator, model object
    :return score: accuracy score of selected estimator, float
    """
    # initialize the Pipeline
    pipe = Pipeline([('estimator', None)])
    rscv = RandomizedSearchCV(pipe,
                              param_distributions=estimator_list,
                              scoring='accuracy',
                              n_iter=max_iter,
                              random_state=18)
    rscv.fit(X_train, y_train)
    model = rscv.best_estimator_[0]
    score = rscv.best_score_
    return [model, score]


def select_best_parameters(model, hyperparams, X_train, y_train):
    """
    Function to select the best parameters based on Randomized Search Cross Validation on selected model
    :param model: selected best estimator, model object
    :param hyperparams: dictionary of hyperparameter values, dictionary
    :param X_train: training data, DataFrame
    :param y_train: training target, DataFrame
    :return best_model: optimized model with selected parameters, model object
    """
    # define parameter grid
    if 'LogisticRegression' in str(type(model)):
        parameter_grid = hyperparams['logistic_regression_parameters']
    elif 'KNeighborsClassifier' in str(type(model)):
        parameter_grid = hyperparams['knn_parameters']
    elif 'DecisionTreeClassifier' in str(type(model)):
        parameter_grid = hyperparams['decision_tree_parameters']
    elif 'SVC' in str(type(model)):
        parameter_grid = hyperparams['svc_parameters']

    # use Grid Search to optimize parameters
    gscv = GridSearchCV(estimator=model,
                        param_grid=parameter_grid,
                        scoring='accuracy')
    gscv.fit(X_train, y_train)
    best_model = gscv.best_estimator_
    return best_model


def evaluate_model(model, X_train, y_train, X_test, y_test):
    """
    Function to train and evaluate a model
    :param model: optimized model with selected parameters, model object
    :param X_train: training data, DataFrame
    :param y_train: training target, DataFrame
    :param X_test: testing data, DataFrame
    :param y_test: testing target, DataFrame
    """
    # train the model
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)

    # define confusion matrix and classification report
    cm = confusion_matrix(y_test, prediction)
    class_rep = classification_report(y_test, prediction)

    # display confusion matrix
    cm_disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Candidate', 'False Positive'])
    fig, axes = plt.subplots()
    cm_disp.plot(ax=axes)
    axes.set(title='Confusion Matrix for Exoplanet Candidate Classification Model')
    fig.tight_layout()
    fig.savefig('Exoplanet_confusion_matrix.png')

    # display classification report
    print(class_rep)


if __name__ == '__main__':
    main()
