# imports
## data manipulation
import pandas as pd
import numpy as np

## data visualisation
import matplotlib.pyplot as plt
import seaborn as sns

## sklearn imports
from sklearn.metrics import roc_auc_score, recall_score, precision_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score, recall_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline

# XGBoost
from xgboost import XGBClassifier

## SMOTE for balancing data
from imblearn.over_sampling import SMOTE

## time
import time


def load_data(data_path, verbose=True, random_state=11, test_size=0.2):
    '''load, transform, and train/test split'''

    data = pd.read_csv(filepath_or_buffer=data_path, sep=';')
    # Drop the dublicates
    data.drop_duplicates(inplace=True) 
    # reset indexs
    data.reset_index(inplace=True, drop=True)

    if verbose:
        print('Raw data')
        print(data.head())

    # Replacing values with binary ()
    data.contact = data.contact.map({'cellular': 1, 'telephone': 0}).astype('uint8') 
    data.loan = data.loan.map({'yes': 1, 'unknown': 0, 'no' : 0}).astype('uint8')
    data.housing = data.housing.map({'yes': 1, 'unknown': 0, 'no' : 0}).astype('uint8')
    data.default = data.default.map({'no': 1, 'unknown': 0, 'yes': 0}).astype('uint8')
    data.pdays = data.pdays.replace(999, 0) # replace with 0 if not contact 
    data.previous = data.previous.apply(lambda x: 1 if x > 0 else 0).astype('uint8') # binary has contact or not

    # binary if there was an outcome of marketing campane
    data.poutcome = data.poutcome.map({'nonexistent':0, 'failure':0, 'success':1}).astype('uint8') 

    # change the sign (we want all be positive values)
    data['cons.conf.idx'] = data['cons.conf.idx'] * -1

    # re-scale variables
    data['nr.employed'] = np.log(data['nr.employed'])
    data.age = np.log(data.age)                              # this can be converted to bins
    data['cons.price.idx'] = np.log(data['cons.price.idx'])

    # less space
    data.euribor3m = data.euribor3m.astype('uint8')
    data.campaign = data.campaign.astype('uint8')
    data.pdays = data.pdays.astype('uint8')

    # Convert target variable into numeric
    data.y = data.y.map({'no':0, 'yes':1}).astype('uint8')

    # target encoding
    data['job_te'] = target_encoder(df=data, column='job', target='y',method='mean')
    data['marital_te'] = target_encoder(df=data, column='marital', target='y',method='mean')
    data['education_te'] = target_encoder(df=data, column='education', target='y',method='mean')

    # data = duration_to_bin(data)

    # cycilcal feature encoding
    monthDict = {'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4,'may': 5, 'jun': 6, 'jul': 7, 'aug': 8, 'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12}
    data['month_sin'] = data.month.map(monthDict).apply(lambda x: np.sin(x / 12 * 2 * np.pi))
    data['month_cos'] = data.month.map(monthDict).apply(lambda x: np.cos(x / 12 * 2 * np.pi))

    dayofweekDict = {'mon':1, 'tue':2, 'wed':3, 'thu':4, 'fri':5, 'sat':6, 'sun':7}
    data['day_sin'] = data.day_of_week.map(dayofweekDict).apply(lambda x: np.sin(x / 7 * 2 * np.pi))
    data['day_cos'] = data.day_of_week.map(dayofweekDict).apply(lambda x: np.cos(x / 7 * 2 * np.pi))    

    if verbose:
        print('Transformed/Cleaned data')
        print(data.head())

    X = data.drop(['job', 'month', 'day_of_week', 'marital', 'education', 'duration', 'y'], axis=1)
    y = data.y

    # set global random state
    random_state = random_state

    # change the order of columns
    cat_cols = ['default','housing','loan','contact','previous','poutcome','marital_te',
                'education_te','job_te','month_sin','month_cos','day_sin','day_cos']
    num_cols = ['age','emp.var.rate','cons.conf.idx','euribor3m','nr.employed','campaign','pdays','cons.price.idx']
    X = X[num_cols + cat_cols]

    # split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    if verbose:
        print('check the shape of splitted train and test sets', X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    X_train.reset_index(inplace=True, drop=True)
    X_test.reset_index(inplace=True, drop=True)
    y_train.reset_index(inplace=True, drop=True)
    y_test.reset_index(inplace=True, drop=True)

    return X_train, y_train, X_test, y_test

def scale_data(X_train, y_train, X_test):
    '''scale the train and test data'''

    cat_cols = ['default','housing','loan','contact','previous','poutcome','marital_te',
                'education_te','job_te','month_sin','month_cos','day_sin','day_cos']
    num_cols = ['age','emp.var.rate','cons.conf.idx','euribor3m','nr.employed','campaign','pdays','cons.price.idx']

    scaler = StandardScaler()
    X_train_num = X_train[num_cols].copy()
    X_test_num = X_test[num_cols].copy()

    scaler.fit(X_train_num)
    X_train_num_scaled = scaler.transform(X_train_num)
    X_test_num_scaled = scaler.transform(X_test_num)

    X_train_num_scaled = pd.DataFrame(X_train_num_scaled, columns=X_train_num.columns)
    X_test_num_scaled = pd.DataFrame(X_test_num_scaled, columns=X_test_num.columns)

    X_train_scaled = X_train_num_scaled.join(X_train[cat_cols])
    X_test_scaled = X_test_num_scaled.join(X_test[cat_cols])

    # balace train data
    smote = SMOTE(random_state=0)
    X_train_balanced, y_balanced = smote.fit_resample(X_train_scaled, y_train)

    return X_train_scaled, X_test_scaled, X_train_balanced, y_balanced

def bar_plot(data, columns, target='y', figsize=(8,4)):
    '''return bar plots for all given columns
    Parameters
    ----------
    data: DataFrame
        dataframe including the given columns
    columns: List
        a list of column names for bar plots
    target: string
        target value, default: 'y'
    figsize: Tuple(x,y)
        size of the plots, default: (8,4)

    Returns
    -------
    None
    '''

    
    for column in columns:
        fig, axes = plt.subplots(nrows=1, ncols=2, figsize=figsize)
        
        # temp df 
        temp_1 = pd.DataFrame()
        # count categorical values
        temp_1['No_deposit'] = data[data[target] == 'no'][column].value_counts()
        temp_1['Yes_deposit'] = data[data[target] == 'yes'][column].value_counts()
        
        temp_1.plot(kind='bar',  ax=axes[0])
        axes[0].set_title('All')
        axes[0].set_ylabel('Number of clients')
        axes[0].set_xlabel(f'{column}')

        # prepare df for proportion plot
        parm_counts = data.groupby([column, target]).size()
        parm_totals = data.groupby(column).size()
        proportions = parm_counts[:, 'yes']/parm_totals

        # Draw bar graph
        plt.style.use('default')

        # plot bars
        proportions.sort_values(ascending=False).plot(kind='bar',  ax=axes[1])
        axes[1].set_title('Campaign Success Proportion by {}'.format(column), fontsize=10)
        axes[1].set_ylabel('Proportion')
        axes[1].set_xlabel(f'{column}')

        fig.suptitle('Distribution and Proportional plot of {} and deposit'.format(column), fontsize=10)

        plt.tight_layout()
        plt.show()

def encode(data, col):
    '''
    one-hot encoding a column of a data frame and return the concat result
    '''
    return pd.concat([data, pd.get_dummies(col, prefix=col.name)], axis=1)

def duration_to_bin(data):
    '''Convert Duration of Call into 5 category'''

    data.loc[data['duration'] <= 102, 'duration'] = 1
    data.loc[(data['duration'] > 102) & (data['duration'] <= 180)  , 'duration'] = 2
    data.loc[(data['duration'] > 180) & (data['duration'] <= 319)  , 'duration'] = 3
    data.loc[(data['duration'] > 319) & (data['duration'] <= 645), 'duration'] = 4
    data.loc[data['duration']  > 645, 'duration'] = 5
    return data

def age_to_bin(data):
    data.loc[data['age'] <= 102, 'duration'] = 1
    data.loc[(data['duration'] > 102) & (data['duration'] <= 180)  , 'duration'] = 2
    data.loc[(data['duration'] > 180) & (data['duration'] <= 319)  , 'duration'] = 3
    data.loc[(data['duration'] > 319) & (data['duration'] <= 645), 'duration'] = 4
    data.loc[data['duration']  > 645, 'duration'] = 5
    return data

def target_encoder(df, column, target, index=None, method='mean'):
    """
    Target-based encoding is numerization of a categorical variables via the target variable. Main purpose is to deal
    with high cardinality categorical features without exploding dimensionality. This replaces the categorical variable
    with just one new numerical variable. Each category or level of the categorical variable is represented by a
    summary statistic of the target for that level.
    Args:
        df (pandas df): Pandas DataFrame containing the categorical column and target.
        column (str): Categorical variable column to be encoded.
        target (str): Target on which to encode.
        index (arr): Can be supplied to use targets only from the train index. Avoids data leakage from the test fold
        method (str): Summary statistic of the target. Mean, median or std. deviation.
    Returns:
        arr: Encoded categorical column.
    """

    index = df.index if index is None else index # Encode the entire input df if no specific indices is supplied

    if method == 'mean':
        encoded_column = df[column].map(df.iloc[index].groupby(column)[target].mean())
    elif method == 'median':
        encoded_column = df[column].map(df.iloc[index].groupby(column)[target].median())
    elif method == 'std':
        encoded_column = df[column].map(df.iloc[index].groupby(column)[target].std())
    else:
        raise ValueError("Incorrect method supplied: '{}'. Must be one of 'mean', 'median', 'std'".format(method))

    return encoded_column

def prepare_grid_list(grid_params_lr, grid_params_rf, grid_params_knn, grid_params_xgb, scoring='accuracy', random_state=11, n_splits=5):
    '''prepare the gridsearch objects for all models'''

    ## Build pipline of classifiers
    # set all CPU
    n_jobs = -1
    # LogisticRegression
    pipe_lr = Pipeline([('lr', LogisticRegression(random_state=random_state, class_weight="balanced", n_jobs=n_jobs, max_iter=1000))])
    # RandomForestClassifier
    pipe_rf = Pipeline([('rf', RandomForestClassifier(random_state=random_state, class_weight='balanced', n_jobs=n_jobs))])
    # KNeighborsClassifier
    pipe_knn = Pipeline([('knn', KNeighborsClassifier(n_jobs=n_jobs))])
    # XGBClassifier
    pipe_xgb = Pipeline([('xgb', XGBClassifier(seed=random_state,n_jobs=n_jobs))])

    ## Set parameters for Grid Search
    # set number pf splits 
    cv = StratifiedKFold(shuffle=True, n_splits=n_splits, random_state=random_state)

    ## Grid search objects
    # for LogisticRegression
    gs_lr = GridSearchCV(pipe_lr, param_grid=grid_params_lr,
                        scoring=scoring, cv=cv) 
    # for RandomForestClassifier
    gs_rf = GridSearchCV(pipe_rf, param_grid=grid_params_rf,
                        scoring=scoring, cv=cv)
    # for KNeighborsClassifier
    gs_knn = GridSearchCV(pipe_knn, param_grid=grid_params_knn,
                        scoring=scoring, cv=cv)
    # for XGBClassifier
    gs_xgb = GridSearchCV(pipe_xgb, param_grid=grid_params_xgb,
                        scoring=scoring, cv=cv)
    
    # models that we iterate over
    grid_list = [gs_lr, gs_rf, gs_knn, gs_xgb]
    return grid_list

def fit_model(grid_list, X_train, y_train, X_test, y_test):
    model_dict = {0:'Logistic', 1:'RandomForest', 2:'KNN', 3:'XGB'}

    # set empty dicts and list
    result_acc = {}
    result_auc = {}
    result_rec = {}
    result_prc = {}
    models = {}

    for index, model in enumerate(grid_list):
        start = time.time()
        print()
        print('+++++++ Start New Model ++++++++++++++++++++++')
        print('Estimator is {}'.format(model_dict[index]))
        model.fit(X_train, y_train)
        print('---------------------------------------------')
        print('best params {}'.format(model.best_params_))
        print('best score is {}'.format(round(model.best_score_,3)))

        auc = round(roc_auc_score(y_test, model.predict_proba(X_test)[:,1]), 3)
        rec = round(recall_score(y_test, model.predict(X_test)), 3)
        prc = round(precision_score(y_test, model.predict(X_test)), 3)
        score = round(model.score(X_test, y_test), 3)

        print('---------------------------------------------')
        print('ROC_AUC: {}, recall: {}, precision: {}, accuracy: {}'.format(auc, rec, prc, score))
        end = time.time()
        print('It lasted for {} sec'.format(round(end - start, 3)))
        print('++++++++ End Model +++++++++++++++++++++++++++')
        print()
        print()
        models[index] = model.best_estimator_
        result_acc[index] = model.best_score_
        result_auc[index] = auc
        result_rec[index] = rec
        result_prc[index] = prc
    return result_acc, result_auc, result_rec, result_prc, models

def plot_feature_importance(model, X_train):
    data = pd.DataFrame(model.feature_importances_, X_train.columns, columns=["feature"])
    data = data.sort_values(by='feature', ascending=False).reset_index()
    sns.barplot(x='index', y='feature', data=data[:20], palette="Blues_d")
    plt.title('Feature inportance of Random Forest after Grid Search')
    plt.xticks(rotation=45)
    plt.show()

def plot_results(result_acc,result_auc,result_rec,result_prc):
    model_dict = {0:'Logistic', 1:'RandomForest', 2:'KNN', 3:'XGB'}

    plt.plot(list(model_dict.values()), list(result_acc.values()), c='r')
    plt.plot(list(model_dict.values()), list(result_auc.values()), c='b')
    plt.plot(list(model_dict.values()), list(result_rec.values()), c='g')
    plt.plot(list(model_dict.values()), list(result_prc.values()), c='black')


    plt.xlabel('Models')
    plt.xticks(rotation=45)
    plt.ylabel('Acc, AUC, REC, PRC')
    plt.title('Result of Grid Search')
    plt.legend(['Acc', 'AUC', 'REC', 'PRC'])
    plt.show()