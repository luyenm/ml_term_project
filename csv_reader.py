import pandas as pd
import numpy as np
import data_visualizer as dv


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

y_axis = training_data.loc[:, 'ClaimAmount']
x_axis = training_data.drop("ClaimAmount", axis=1, inplace=False)

for first_feature in x_axis.keys().tolist():
    for second_feature in x_axis.keys().tolist():
        if first_feature != second_feature:
            dv.generate_scatterplot(x_axis.loc[:, second_feature], x_axis.loc[:, first_feature], second_feature, first_feature)
