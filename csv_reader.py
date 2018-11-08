import pandas as pd
import numpy as np
import data_visualizer as dv
import os


'''
Current insights about the data:
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
training_data = pd.read_csv('trainingset.csv')
if not os.path.exists('scatterplot'):
    os.makedirs('scatterplot')
if not os.path.exists('histogram'):
    os.makedirs('histogram')

not_claimed = training_data[training_data.ClaimAmount == 0]
claimed_money = training_data[training_data.ClaimAmount != 0]

print('Not claim: ', len(not_claimed), '\t\tClaim Money: ', len(claimed_money))

x_claimed_money = claimed_money.drop('ClaimAmount', axis=1, inplace=False)
y_claimed_money = claimed_money.loc[:, 'ClaimAmount']
x_not_claimed = not_claimed.drop('ClaimAmount', axis=1, inplace=False)
y_not_claimed = not_claimed.loc[:, 'ClaimAmount']

for first_feature in training_data.keys().tolist():
    for second_feature in training_data.keys().tolist():
        if first_feature != second_feature:
            dv.image_scatterplot(x_claimed_money.loc[:, second_feature], y_claimed_money.loc[:, first_feature], second_feature, first_feature, )
            dv.image_scatterplot(x_not_claimed.loc[:, second_feature], y_not_claimed.loc[:, first_feature], second_feature, first_feature, )

for feature in training_data.keys().tolist():
    dv.image_histogram(not_claimed.loc[:, feature], feature, 'not_claimed_')
    dv.image_histogram(claimed_money.loc[:, feature], feature, 'claimed_money_')


