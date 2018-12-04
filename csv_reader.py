import pandas as pd
from pathlib import Path
import os
from sklearn import preprocessing


'''
Current insights about the data:
ASSUMPTIONS:
Anything below 26 unique values is assume to be categorical.

# rowIndex - ID of person?
# feature1	(Values are between 0.001 and 1.000)
# feature2	(Values are between 9 and 940)
# feature3	(Values are 0 .. 8)
# feature4	(Values are 0, 1) - Categorical
# feature5	(Values are 0, 1) - Categorical
# feature6	(Values are between 0 and 51)
# feature7	(Values are 0 .. 3)
# feature8	(Values are between 18 and 103) Age?
# feature9	(Values are 0, 1) - Categorical
# feature10	(Values are 50 .. 272) Height?
# feature11	(Values are 0 .. 8) 
# feature12	(Values are 0 .. 26)
# feature13	(Values are 0 .. 5)
# feature14	(Values are 0 .. 3)
# feature15	(Values are 0 .. 9)
# feature16	(Values are 0 .. 5)
# feature17	(Values are 1 .. 20)
# feature18	(Values are 0, 1, 2)


# ClaimAmount - Dollar amount for claims that are made ($1 or $1000)
'''
# TODO: Implement more data pre-processing for categorical data.


def onehot(dataframe):
    TEXT = 'onehot|'
    print(TEXT, 'Making onehot')
    NEW_DATA = {'feature3': range(0,9), 'feature6': range(0, 52),
     'feature7': range(0, 4), 'feature11': range(0, 9),
     'feature12': range(0, 27), 'feature13': range(0, 6),
     'feature14': range(0, 4), 'feature15': range(0, 10),
     'feature16': range(0, 6), 'feature17': range(1, 21),
     'feature18': range(0, 3), 'feature8': range(10, 101, 10)
     }
    new_dataframe = pd.DataFrame()
    for key in dataframe.keys().values:
        if key not in NEW_DATA.keys():
            #print(TEXT, 'add key', key)
            new_dataframe[key] = dataframe.loc[:,key]
        elif key == NEW_DATA['feature8']:
            col = dataframe.loc[:, key]
            col = col.values
            for increment in NEW_DATA.get(key):
                new_col = []
                for row in col: new_col.append(1 if increment + 10 > row >= increment else 0)
                label = key + '_' + increment.__str__() + '-' + (increment+10).__str__()
                new_dataframe[label] = new_col
        else:
            col = dataframe.loc[:,key]
            col = col.values
            for increment in NEW_DATA.get(key):
                new_col = []
                for row in col: new_col.append(1 if row == increment else 0)
                label = key+'_'+increment.__str__()
                #print(TEXT, 'add key', label)
                new_dataframe[label] = new_col
    return new_dataframe


def normalization(dataframe):
    weighted = dataframe.values
    min_max = preprocessing.MinMaxScaler()
    scaled = min_max.fit_transform(weighted)
    norm_frame = pd.DataFrame(scaled)
    norm_frame.columns = dataframe.columns.values
    return norm_frame


competition_set = None
competition_set_path = Path('competitionset.csv')
if competition_set_path.exists():
    competition_set = pd.read_csv('competitionset.csv')
    competition_set.drop('rowIndex', axis=1, inplace=True)


training_data = pd.read_csv('trainingset.csv')
# training_data = training_data.sample(frac=1).reset_index(drop=True)
training_data.drop('rowIndex', axis=1, inplace=True)
test_data = pd.read_csv('testset.csv')
test_data.drop('rowIndex', axis=1, inplace=True)


if not os.path.exists('scatterplot'):
    os.makedirs('scatterplot')
if not os.path.exists('histogram'):
    os.makedirs('histogram')

not_claimed = training_data[training_data.ClaimAmount == 0]
claimed_money = training_data[training_data.ClaimAmount != 0]

not_claimed_categorical = onehot(not_claimed)
claimed_money_categorical = onehot(claimed_money)
training_data_categorical = onehot(training_data)

print('Not claim: ', len(not_claimed), '\t\tClaimed Money: ', len(claimed_money))

x_training_data = training_data.drop('ClaimAmount', axis=1, inplace=False)
y_training_data = training_data.loc[:, 'ClaimAmount']
x_training_data_categorical = training_data_categorical.drop('ClaimAmount', axis=1, inplace=False)
y_training_data_categorical = training_data_categorical.loc[:, 'ClaimAmount']


x_claimed_money = claimed_money.drop('ClaimAmount', axis=1, inplace=False)
y_claimed_money = claimed_money.loc[:, 'ClaimAmount']
x_not_claimed = not_claimed.drop('ClaimAmount', axis=1, inplace=False)
y_not_claimed = not_claimed.loc[:, 'ClaimAmount']

x_claimed_money_categorical = claimed_money_categorical.drop('ClaimAmount', axis=1, inplace=False)
y_claimed_money_categorical = claimed_money_categorical.loc[:, 'ClaimAmount']
x_not_claimed_categorical = not_claimed.drop('ClaimAmount', axis=1, inplace=False)
y_not_claimed_categorical = not_claimed.loc[:, 'ClaimAmount']


def get_dataset():
    return x_training_data, y_training_data


def get_dataset_categorical():
    return x_training_data_categorical, y_training_data_categorical


def get_claims():
    return x_claimed_money, y_claimed_money


def get_claims_categorical():
    return x_claimed_money_categorical, y_claimed_money_categorical


def get_unclaimed():
    return x_not_claimed, y_not_claimed


def get_trainset():
    return training_data


def get_testset():
    return test_data


def get_testset_categorical():
    return onehot(test_data)


def get_competitive_set_categorical():
    return onehot(competition_set)