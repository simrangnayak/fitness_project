# Import packages
from tracemalloc import start
from typing import final
import pandas as pd
import xml.etree.ElementTree as ET
import numpy as np
import plotly.express as px
import io
import matplotlib.pyplot as plt 
import seaborn as sb 
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn import metrics 
from sklearn.svm import SVC 
from xgboost import XGBRegressor 
from sklearn.linear_model import LinearRegression, Lasso, Ridge 
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error as mae 

pd.set_option('display.max_columns', 500)

# Load data
path = 'export.xml'

# Important Variables
start_date = '2023-03-19' # day I started wearing Apple Watch
min_counts = 0 # only use workouts with data more than min_counts records
record_corr = 0.1 # minimum correlation with BasalEnergyBurned for records
workout_corr = 0.1 # minimum correlation with ActiveEnergyBurned for workouts

# Create element tree object
tree = ET.parse(path)
root = tree.getroot()

# Extract attributes for every health record
record_list = [x.attrib for x in root.iter('Record')]
record_data = pd.DataFrame(record_list) # change to dataframe

# Clean up Dataframe
record_data['Day'] = pd.to_datetime(record_data['startDate']).dt.weekday # extract day of startdate, turn to numeric
record_data['Date'] = pd.to_datetime(record_data['startDate']).dt.strftime('%Y-%m-%d') # extract date of startdate
record_data = record_data[record_data['Date'] >= start_date]
record_data['value'] = pd.to_numeric(record_data['value'], errors='coerce') # change value column to numeric
record_data['value'] = record_data['value'].fillna(1.0) # if NaN then change to occurence (1.0)
record_data['type'] = record_data['type'].str.replace('HKQuantityTypeIdentifier', '') # shorten observation name
record_data['type'] = record_data['type'].str.replace('HKCategoryTypeIdentifier', '')

# Remove unnecessary columns (remove after cleaning as you need startDate to create Date column)
remove_cols = ['sourceName', 'sourceVersion', 'device', 'startDate', 'creationDate', 'endDate']
record_data = record_data.drop(remove_cols, axis=1)

# Records to use: StepCount, DistanceWalkingRunnng, BasalEnergyBurned, AppleExerciseTime, RestingHeartRate, WalkingHeartRateAverage, AppleStandTime, HeartRateVariabilitySDNN

# Select records to use and aggregate values daily (by either sum or avg)
sum_record_list = ['StepCount', 'BasalEnergyBurned', 'AppleExerciseTime', 'AppleStandTime', 'ActiveEnergyBurned', 'DistanceWalkingRunning'] # columns to aggregate by sum
record_data_sum = record_data[record_data['type'].isin(sum_record_list)]
record_data_sum = record_data_sum.groupby(['type', 'Date', 'Day', 'unit'], as_index=False).agg({'value': 'sum'})

avg_record_list = ['HeartRateVariabilitySDNN', 'RestingHeartRate', 'WalkingHeartRateAverage'] # columns to aggregate by avg
record_data_avg = record_data[record_data['type'].isin(avg_record_list)]
record_data_avg = record_data_avg.groupby(['type', 'Date', 'Day', 'unit'], as_index=False).agg({'value': 'mean'})

daily_df = pd.concat([record_data_sum, record_data_avg])
daily_df = daily_df.sort_values(['Date', 'type']) # sort by date and type of record
daily_df = daily_df[['Date', 'Day', 'type', 'value']] # rearranging columns and removing unit

# Pivot Table to reframe records as columns
daily_df = daily_df.pivot(index=['Date', 'Day'], columns='type', values='value')
daily_df = daily_df.rename_axis(None, axis=1).reset_index() # remove multi-indexing
daily_df = daily_df.dropna(axis=0) # remove rows where you have missing data

'''
# Check relationship between continuous features and target variables
continuous_features = ['StepCount', 'AppleExerciseTime', 'AppleStandTime', 'RestingHeartRate', 'ActiveEnergyBurned']
plt.subplots(figsize=(15, 10)) 
for i, f_col in enumerate(continuous_features):
    plt.subplot(2, 3, i + 1)
    sb.scatterplot(x=f_col, y='BasalEnergyBurned', data=daily_df)
plt.tight_layout()
plt.show()

# Check density of continous features
for i, f_col in enumerate(continuous_features):
    plt.subplot(2, 3, i + 1)
    sb.histplot(data=daily_df, x=f_col, kde=True, stat='density')
plt.tight_layout()
plt.show()

# Check correlation heatmap for feature selection.
# This is to check if features are correlated with the target variable and if any features are correlated. This helps with feature reduction
# If features are NOT highly correlated with the target variable, the feature does not have useful relationship and will not help with prediction.
# If features are correlated, they are redundant as we can predict one from the other. Having both just increases complexity of the algorithm unnecessarily.
# RestingHeartRate and WalkingHeartRate are not as correlated with BasalEnergyBurned, so removing those features.
# DistanceWalkingRunning and StepCount are correlated, so only using StepCount.
plt.figure(figsize=(10, 10)) 
sb.heatmap(daily_df.corr(), annot=True, cbar=True) 
plt.show()
'''

# Optional: Feature selection where correlation with BasalEnergyBurned is less than workout_corr
corr_values = daily_df.corr()['BasalEnergyBurned'].abs()
target_corr = corr_values.index[corr_values <= record_corr].tolist()

# Calculate Basal Energy Burned here -> add active calories burned here from exercise
redundant_var = ['DistanceWalkingRunning']
features = daily_df.drop(['Date', 'BasalEnergyBurned'] + target_corr + redundant_var, axis=1) 

X_train, X_val, Y_train, Y_val = train_test_split(features, daily_df['BasalEnergyBurned'], test_size=0.40, random_state=22) 
scaler = StandardScaler() 
X_train = scaler.fit_transform(X_train) 
X_val = scaler.transform(X_val)
    
models = [LinearRegression(), XGBRegressor(), Lasso(), RandomForestRegressor(), Ridge()]
print('---------------------------------------')
print('Basal Energy Expenditure') 
print()
for j in range(5): 
    models[j].fit(X_train, Y_train)
    print(f'{models[j]} : ')
    train_preds = models[j].predict(X_train)
    print('Training Error : ', mae(Y_train, train_preds))
        
    val_preds = models[j].predict(X_val)
    print('Validation Error : ', mae(Y_val, val_preds))
    print()

# Workout Data
workout_list = list(root.iter('Workout')) # extract attributes for each workout

final_workout_dict = []
for i in range(len(workout_list)): # iterating through each workout
    workout_dict = workout_list[i].attrib 
    average_list = ['HeartRate', 'RunningSpeed', 'RunningPower', 'RunningGroundContactTime', 'RunningVerticalOscillation', 'RunningStrideLength']
    
    WorkoutStatisticsList = list(workout_list[i].iter('WorkoutStatistics')) # extract statistics for workout i
    for j, WorkoutStatistics in enumerate(WorkoutStatisticsList): # iterating through workout i's statistics
        type = WorkoutStatistics.attrib['type'].replace('HKQuantityTypeIdentifier', '')

        # workout metrics in WorkoutStatistics are two types: those that have an 'average' data value and a 'sum' value
        # the following if-else statement uses average_list to determine if we need to grab 'average' or 'sum'
        if type in average_list:
            workout_dict[type] = WorkoutStatistics.attrib['average']
        else:
            workout_dict[type] = WorkoutStatistics.attrib['sum']
    final_workout_dict.append(workout_dict) # appending workout i statistics to final_workout_dict 

final_workout_df = pd.DataFrame(final_workout_dict) # create final_workout_df dataframe

# Clean up Dataframe (create date and clean up activity name)
final_workout_df['Date'] = pd.to_datetime(final_workout_df['startDate']).dt.strftime('%Y-%m-%d')
final_workout_df['workoutActivityType'] = final_workout_df['workoutActivityType'].str.replace('HKWorkoutActivityType', '')

# Remove unnecessary columns (remove after cleaning as you need startDate to create Date column)
remove_cols = ['sourceName', 'sourceVersion', 'device', 'creationDate', 'startDate', 'endDate', 'durationUnit', 'StepCount', 'RunningSpeed', 
               'RunningPower', 'RunningGroundContactTime', 'RunningVerticalOscillation', 'RunningStrideLength', 'DistanceSwimming', 
               'SwimmingStrokeCount', 'DistanceCycling']
final_workout_df = final_workout_df.drop(remove_cols, axis=1)

# Transform necessary columns into float type
final_workout_df['duration'] = final_workout_df['duration'].astype(float)
final_workout_df['BasalEnergyBurned'] = final_workout_df['BasalEnergyBurned'].astype(float)
final_workout_df['ActiveEnergyBurned'] = final_workout_df['ActiveEnergyBurned'].astype(float)
final_workout_df['HeartRate'] = final_workout_df['HeartRate'].astype(float)
final_workout_df['DistanceWalkingRunning'] = final_workout_df['DistanceWalkingRunning'].astype(float)

# Optional: Remove workouts with counts <= min_counts
workout_cnts = final_workout_df.groupby(['workoutActivityType'])['workoutActivityType'].count().reset_index(name='count')
workout_cnts = workout_cnts[workout_cnts['count'] <= min_counts]
workout_cnts = workout_cnts['workoutActivityType'].values.tolist()
final_workout_df = final_workout_df[~final_workout_df['workoutActivityType'].isin(workout_cnts)]

# One-hot encoding workoutactivityType category
one_hot = pd.get_dummies(final_workout_df['workoutActivityType'])
final_workout_df = final_workout_df.join(one_hot)
final_workout_df = final_workout_df.drop(['workoutActivityType'], axis=1) # remove categorical variables after one-hot encoding

# Sort Data and fill NaN values
final_workout_df = final_workout_df.sort_values(['Date'])
final_workout_df = final_workout_df.fillna(0) # fill column DistanceWalkingRunning with 0 when not running or walking

'''
# Check relationship between continuous features and target variables
continuous_features = ['duration', 'HeartRate', 'DistanceWalkingRunning', 'BasalEnergyBurned']
plt.subplots(figsize=(15, 10)) 
for i, f_col in enumerate(continuous_features):
    plt.subplot(2, 2, i + 1)
    sb.scatterplot(x=f_col, y='ActiveEnergyBurned', data=final_workout_df)
plt.tight_layout()
plt.show()

# Check density of continous features
for i, f_col in enumerate(continuous_features):
    plt.subplot(2, 2, i + 1)
    sb.histplot(data=final_workout_df, x=f_col, kde=True, stat='density')
plt.tight_layout()
plt.show()

# Check correlation heatmap for feature selection.
# This is to check if features are correlated with the target variable and if any features are correlated. This helps with feature reduction
# If features are NOT highly correlated with the target variable, the feature does not have useful relationship and will not help with prediction.
# If features are correlated, they are redundant as we can predict one from the other. Having both just increases complexity of the algorithm unnecessarily.
plt.figure(figsize=(10, 10)) 
sb.heatmap(final_workout_df.corr(), annot=True, cbar=True)
plt.show()
'''

# Optional: Feature selection where correlation with ActiveEnergyBurned is less than workout_corr
corr_values = final_workout_df.corr()['ActiveEnergyBurned'].abs()
target_corr = corr_values.index[corr_values <= workout_corr].tolist()

# Model Training - calculate active energy burned during workout here
redundant_var = ['duration']
features = final_workout_df.drop(['Date', 'ActiveEnergyBurned'] + redundant_var + target_corr, axis=1) 

X_train, X_val, Y_train, Y_val = train_test_split(features, final_workout_df['ActiveEnergyBurned'], test_size=0.3, random_state=22) 
scaler = StandardScaler() 
X_train = scaler.fit_transform(X_train) 
X_val = scaler.transform(X_val)

models = [LinearRegression(), XGBRegressor(), Lasso(), RandomForestRegressor(), Ridge()]
print('---------------------------------------')
print('Active Workout Energy Expenditure') 
print()   
for j in range(5): 
    models[j].fit(X_train, Y_train)
    print(f'{models[j]} : ')
    train_preds = models[j].predict(X_train)
    print('Training Error : ', mae(Y_train, train_preds))
        
    val_preds = models[j].predict(X_val)
    print('Validation Error : ', mae(Y_val, val_preds))
    print() 



'''
# Plotting
# Basal Energy Burned while using Apple Watch
start_date = '2023-03-19' # start date of wearing Apple Watch


basal_energy_daily = record_data_daily[record_data_daily['type'] == 'ActiveEnergyBurned' & record_data_daily['Date'] >= start_date]
fig = px.line(basal_energy_daily, x=basal_energy_daily['Date'], y=basal_energy_daily['value'], markers=True)
fig.update_layout(title_text="Basal Energy Progress Since Wearing Apple Watch")
fig.show()
'''
