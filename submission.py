import numpy as np
import pandas as pd
import math
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error


## Project-Part1
def predict_COVID_part1(svm_model, train_df, train_labels_df, past_cases_interval, past_weather_interval, test_feature):
    tr = train_df[['day', 'max_temp', 'max_dew', 'max_humid', 'dailly_cases']].copy()
    daylst= tr['day'].values.tolist()
    templst = tr['max_temp'].values.tolist()
    dewlst =  tr['max_dew'].values.tolist()
    humidlst= tr['max_humid'].values.tolist()
    caseslst= tr['dailly_cases'].values.tolist()
    temp_list = []
    dew_list = []
    humid_list = [] 
    cases_list = []
    row_list = []
    for i in range(30, len(train_df)):
        temp_list.append(templst[i-30:i])
    new_df_temps = pd.DataFrame(temp_list, columns = ['max_temp-30', 'max_temp-29', 'max_temp-28', 'max_temp-27', 'max_temp-26', 
                                    'max_temp-25', 'max_temp-24', 'max_temp-23', 'max_temp-22', 'max_temp-21', 
                                    'max_temp-20', 'max_temp-19', 'max_temp-18', 'max_temp-17', 'max_temp-16', 
                                    'max_temp-15', 'max_temp-14', 'max_temp-13', 'max_temp-12', 'max_temp-11', 
                                    'max_temp-10', 'max_temp-9', 'max_temp-8', 'max_temp-7', 'max_temp-6', 'max_temp-5', 
                                    'max_temp-4', 'max_temp-3', 'max_temp-2', 'max_temp-1'])
    for i in range(30, len(train_df)):
        dew_list.append(dewlst[i-30:i])
    new_df_dews = pd.DataFrame(dew_list, columns = ['max_dew-30', 'max_dew-29','max_dew-28', 'max_dew-27', 'max_dew-26', 'max_dew-25', 'max_dew-24', 'max_dew-23', 
                                        'max_dew-22', 'max_dew-21', 'max_dew-20', 'max_dew-19', 'max_dew-18', 'max_dew-17', 
                                        'max_dew-16', 'max_dew-15', 'max_dew-14', 'max_dew-13', 'max_dew-12', 'max_dew-11', 
                                        'max_dew-10', 'max_dew-9', 'max_dew-8', 'max_dew-7', 'max_dew-6', 'max_dew-5', 
                                        'max_dew-4', 'max_dew-3', 'max_dew-2', 'max_dew-1'])
    for i in range(30, len(train_df)):
        humid_list.append(humidlst[i-30:i])
    new_df_humids = pd.DataFrame(humid_list, columns = ['max_humid-30', 'max_humid-29','max_humid-28', 'max_humid-27', 'max_humid-26', 'max_humid-25', 'max_humid-24', 
                                        'max_humid-23', 'max_humid-22', 'max_humid-21', 'max_humid-20', 'max_humid-19', 
                                        'max_humid-18', 'max_humid-17', 'max_humid-16', 'max_humid-15', 'max_humid-14', 
                                        'max_humid-13', 'max_humid-12', 'max_humid-11', 'max_humid-10', 'max_humid-9', 
                                        'max_humid-8', 'max_humid-7', 'max_humid-6', 'max_humid-5', 'max_humid-4', 
                                        'max_humid-3', 'max_humid-2', 'max_humid-1'])
    for i in range(30, len(train_df)):
        cases_list.append(caseslst[i-30:i])
    new_df_daillycases= pd.DataFrame(cases_list, columns = ['dailly_cases-30', 'dailly_cases-29', 
                                        'dailly_cases-28', 'dailly_cases-27', 'dailly_cases-26', 'dailly_cases-25', 
                                        'dailly_cases-24', 'dailly_cases-23', 'dailly_cases-22', 'dailly_cases-21', 
                                        'dailly_cases-20', 'dailly_cases-19', 'dailly_cases-18', 'dailly_cases-17', 
                                        'dailly_cases-16', 'dailly_cases-15', 'dailly_cases-14', 'dailly_cases-13', 
                                        'dailly_cases-12', 'dailly_cases-11', 'dailly_cases-10', 'dailly_cases-9', 
                                        'dailly_cases-8', 'dailly_cases-7', 'dailly_cases-6', 'dailly_cases-5', 
                                        'dailly_cases-4', 'dailly_cases-3', 'dailly_cases-2', 'dailly_cases-1'])
    new_df_temps = new_df_temps.iloc[:, 30 - past_weather_interval:]
    new_df_dews = new_df_dews.iloc[:, 30 - past_weather_interval:]
    new_df_humids = new_df_humids.iloc[:, 30 - past_weather_interval:]
    new_df_daillycases = new_df_daillycases.iloc[:, 30 - past_cases_interval:]
    features = pd.merge(new_df_temps, new_df_dews, left_index = True, right_index = True)
    features = pd.merge(features, new_df_humids, left_index = True, right_index = True)
    features = pd.merge(features, new_df_daillycases, left_index = True, right_index = True)
    test_features = test_feature[list(features.columns)]
    test_features = pd.DataFrame(test_features)
    test_features = test_features.T
    feature_labels = train_labels_df[30:]
    feature_labels = feature_labels['dailly_cases']
    svm_model.fit(features, feature_labels)
    pred = svm_model.predict(test_features)
    pred = math.floor(pred)
    return pred


## Project-Part2
def predict_COVID_part2(train_df, train_labels_df, test_feature):
    tr = train_df[['day', 'max_temp', 'max_dew', 'max_humid', 'dailly_cases']].copy()
    daylst= tr['day'].values.tolist()
    templst = tr['max_temp'].values.tolist()
    dewlst =  tr['max_dew'].values.tolist()
    humidlst= tr['max_humid'].values.tolist()
    caseslst= tr['dailly_cases'].values.tolist()
    temp_list = []
    dew_list = []
    humid_list = [] 
    cases_list = []
    row_list = []
    for i in range(30, len(train_df)):
        temp_list.append(templst[i-30:i])
    new_df_temps = pd.DataFrame(temp_list, columns = ['max_temp-30', 'max_temp-29', 'max_temp-28', 'max_temp-27', 'max_temp-26', 
                                    'max_temp-25', 'max_temp-24', 'max_temp-23', 'max_temp-22', 'max_temp-21', 
                                    'max_temp-20', 'max_temp-19', 'max_temp-18', 'max_temp-17', 'max_temp-16', 
                                    'max_temp-15', 'max_temp-14', 'max_temp-13', 'max_temp-12', 'max_temp-11', 
                                    'max_temp-10', 'max_temp-9', 'max_temp-8', 'max_temp-7', 'max_temp-6', 'max_temp-5', 
                                    'max_temp-4', 'max_temp-3', 'max_temp-2', 'max_temp-1'])
    for i in range(30, len(train_df)):
        dew_list.append(dewlst[i-30:i])
    new_df_dews = pd.DataFrame(dew_list, columns = ['max_dew-30', 'max_dew-29','max_dew-28', 'max_dew-27', 'max_dew-26', 'max_dew-25', 'max_dew-24', 'max_dew-23', 
                                        'max_dew-22', 'max_dew-21', 'max_dew-20', 'max_dew-19', 'max_dew-18', 'max_dew-17', 
                                        'max_dew-16', 'max_dew-15', 'max_dew-14', 'max_dew-13', 'max_dew-12', 'max_dew-11', 
                                        'max_dew-10', 'max_dew-9', 'max_dew-8', 'max_dew-7', 'max_dew-6', 'max_dew-5', 
                                        'max_dew-4', 'max_dew-3', 'max_dew-2', 'max_dew-1'])
    for i in range(30, len(train_df)):
        humid_list.append(humidlst[i-30:i])
    new_df_humids = pd.DataFrame(humid_list, columns = ['max_humid-30', 'max_humid-29','max_humid-28', 'max_humid-27', 'max_humid-26', 'max_humid-25', 'max_humid-24', 
                                        'max_humid-23', 'max_humid-22', 'max_humid-21', 'max_humid-20', 'max_humid-19', 
                                        'max_humid-18', 'max_humid-17', 'max_humid-16', 'max_humid-15', 'max_humid-14', 
                                        'max_humid-13', 'max_humid-12', 'max_humid-11', 'max_humid-10', 'max_humid-9', 
                                        'max_humid-8', 'max_humid-7', 'max_humid-6', 'max_humid-5', 'max_humid-4', 
                                        'max_humid-3', 'max_humid-2', 'max_humid-1'])
    for i in range(30, len(train_df)):
        cases_list.append(caseslst[i-30:i])
    new_df_daillycases= pd.DataFrame(cases_list, columns = ['dailly_cases-30', 'dailly_cases-29', 
                                        'dailly_cases-28', 'dailly_cases-27', 'dailly_cases-26', 'dailly_cases-25', 
                                        'dailly_cases-24', 'dailly_cases-23', 'dailly_cases-22', 'dailly_cases-21', 
                                        'dailly_cases-20', 'dailly_cases-19', 'dailly_cases-18', 'dailly_cases-17', 
                                        'dailly_cases-16', 'dailly_cases-15', 'dailly_cases-14', 'dailly_cases-13', 
                                        'dailly_cases-12', 'dailly_cases-11', 'dailly_cases-10', 'dailly_cases-9', 
                                        'dailly_cases-8', 'dailly_cases-7', 'dailly_cases-6', 'dailly_cases-5', 
                                        'dailly_cases-4', 'dailly_cases-3', 'dailly_cases-2', 'dailly_cases-1'])
    new_df_temps = new_df_temps.iloc[:, 30 - past_weather_interval:]
    new_df_dews = new_df_dews.iloc[:, 30 - past_weather_interval:]
    new_df_humids = new_df_humids.iloc[:, 30 - past_weather_interval:]
    new_df_daillycases = new_df_daillycases.iloc[:, 30 - past_cases_interval:]
    features = pd.merge(new_df_temps, new_df_dews, left_index = True, right_index = True)
    features = pd.merge(features, new_df_humids, left_index = True, right_index = True)
    features = pd.merge(features, new_df_daillycases, left_index = True, right_index = True)

    svm_model = SVR()
    svm_model.set_params(**{'kernel': 'rbf', 'degree': 1, 'C': 170000,'gamma': 'scale', 'coef0': 0.1, 'tol': 0.1, 'epsilon': 10})
    
    test_features = test_feature[list(features.columns)]
    test_features = pd.DataFrame(test_features)
    test_features = test_features.T
    feature_labels = train_labels_df[30:]
    feature_labels = feature_labels['dailly_cases']
    svm_model.fit(features, feature_labels)
    pred = svm_model.predict(test_features)
    pred = math.floor(pred)
    return pred 