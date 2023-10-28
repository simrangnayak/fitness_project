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

pd.set_option('display.max_columns', 500)

path = '/Users/simrannayak/Desktop/apple_health_export/export.xml' # load data

# Create element tree object
tree = ET.parse(path)
root = tree.getroot()

# Health Record Data
record_list = [x.attrib for x in root.iter('Record')] # for every health record, extract attributes
record_data = pd.DataFrame(record_list) # change to dataframe
# record_data['duration'] = (pd.to_datetime(record_data['endDate']) - pd.to_datetime(record_data['startDate'])).astype('timedelta64[D]') # duration of record

record_data = record_data.drop(['sourceName', 'sourceVersion', 'device', 'creationDate', 'endDate'], axis=1) # drop unnecessary columns

record_data['Day'] = pd.to_datetime(record_data['startDate']).dt.weekday # extract day of startdate, turn to numeric
record_data['Date'] = pd.to_datetime(record_data['startDate']).dt.strftime('%Y-%m-%d') # extract date of startdate
# record_data['Month'] = pd.to_datetime(record_data['startDate']).dt.strftime('%B') # extract month of startdate

record_data['value'] = pd.to_numeric(record_data['value'], errors='coerce') # change value column to numeric
record_data['value'] = record_data['value'].fillna(1.0) # if NaN then change to occurence (1.0)
record_data['type'] = record_data['type'].str.replace('HKQuantityTypeIdentifier', '') # shorten observation name
record_data['type'] = record_data['type'].str.replace('HKCategoryTypeIdentifier', '')

record_data = record_data[['type', 'Date', 'Day', 'value', 'unit']] # reorder columns

# Daily Health Record
sum_record_list = ['StepCount', 'DistanceWalkingRunning', 'BasalEnergyBurned', 'ActiveEnergyBurned', 'FlightsClimbed', 'AppleExerciseTime',
                   'DistanceCycling', 'DistanceSwimming', 'AppleStandTime'] 
                   # columns to aggregate by avg
record_data_sum = record_data[record_data['type'].isin(sum_record_list)]
record_data_sum = record_data_sum.groupby(['type', 'Date', 'Day', 'unit'], as_index=False).agg({'value': 'sum'})

avg_record_list = ['HeartRateVariabilitySDNN', 'RestingHeartRate', 'WalkingHeartRateAverage', 'HeartRateRecoveryOneMinute']
                   # columns to aggregate by avg
record_data_avg = record_data[record_data['type'].isin(avg_record_list)]
record_data_avg = record_data_avg.groupby(['type', 'Date', 'Day', 'unit'], as_index=False).agg({'value': 'mean'})

record_data_daily = pd.concat([record_data_sum, record_data_avg])
record_data_daily = record_data_daily.sort_values(['Date', 'type']) # sort by date and type of record
record_data_daily = record_data_daily[['Date', 'Day', 'type', 'value']] # rearranging columns and removing unit

record_data_pivot = record_data_daily.pivot(index=['Date', 'Day'], columns='type', values='value') # pivot table
record_data_pivot = record_data_pivot.rename_axis(None, axis=1).reset_index() # remove multi-indexing
record_data_pivot = record_data_pivot.fillna(0)
record_data_pivot = record_data_pivot[['Date', 'Day', 'StepCount', 'FlightsClimbed', 'AppleExerciseTime', 'AppleStandTime', 'RestingHeartRate', 
                                       'WalkingHeartRateAverage', 'HeartRateRecoveryOneMinute', 'HeartRateVariabilitySDNN', 'DistanceWalkingRunning', 
                                       'DistanceCycling', 'DistanceSwimming', 'BasalEnergyBurned', 'ActiveEnergyBurned']] # rearrange columns

# Workout Data
final_workout_dict = [] # create dictionary to add workout statistics in
workout_list = list(root.iter('Workout')) # grab workout data

for i in range(len(workout_list)): # iterating through each workout
    workout_dict = workout_list[i].attrib 
    average_list = ['HeartRate', 'RunningSpeed', 'RunningPower', 'RunningGroundContactTime', 'RunningVerticalOscillation', 'RunningStrideLength']
                    # workout metrics that have an 'average' value
    
    WorkoutStatisticsList = list(workout_list[i].iter('WorkoutStatistics')) # grab statistics for workout i
    for j, WorkoutStatistics in enumerate(WorkoutStatisticsList): # iterating through workout i statistics
        type = WorkoutStatistics.attrib['type'].replace('HKQuantityTypeIdentifier', '') # clean up workout statistics name

        # workout metrics in WorkoutStatistics are two types: those that have an 'average' data value and a 'sum' value
        # the following if-else statement uses average_list to determine if we need to grab 'average' or 'sum'
        if type in average_list:
            workout_dict[type] = WorkoutStatistics.attrib['average']
        else:
            workout_dict[type] = WorkoutStatistics.attrib['sum']
    
    final_workout_dict.append(workout_dict) # appending workout i statistics to final_workout_dict 

final_workout_df = pd.DataFrame(final_workout_dict) # create final_workout_df dataframe

final_workout_df = final_workout_df.drop(['sourceName','sourceVersion', 'device', 'creationDate','endDate'], axis=1) # remove unnecessary columns

final_workout_df['Day'] = pd.to_datetime(final_workout_df['startDate']).dt.weekday # extract day of startdate
final_workout_df['Date'] = pd.to_datetime(final_workout_df['startDate']).dt.strftime('%Y-%m-%d') # extract date of startdate
# final_workout_df['Month'] = pd.to_datetime(final_workout_df['startDate']).dt.strftime('%B') # extract month of startdate

final_workout_df['workoutActivityType'] = final_workout_df['workoutActivityType'].str.replace('HKWorkoutActivityType', '') # clean up activity name
final_workout_df['duration'] = final_workout_df['duration'].astype(float) # transform duration into float
# final_workout_df['ActiveEnergyBurned'] = final_workout_df['ActiveEnergyBurned'].astype(float) # transform energy burnt into float
# final_workout_df['BasalEnergyBurned'] = final_workout_df['BasalEnergyBurned'].astype(float) # transform energy burnt into float
# final_workout_df['HeartRate'] = final_workout_df['HeartRate'].astype(float) # transform heart rate into float

final_workout_df = final_workout_df[['workoutActivityType', 'Date', 'Day', 'duration']] # reorder columns and only use key columns
                                     # columns not used: StepCount, RunningSpeed, RunningPower, RunningGroundContactTime, RunningGroundContactTime, 
                                     # RunningVerticalOscillation, RunningStrideLength, DistanceSwimming, SwimmingStrokeCount, DistanceCycling, 
                                     # DistanceWalkingRunning, ActiveEnergyBurned, BasalEnergyBurned, HeartRate, durationUnit
final_workout_df = final_workout_df.sort_values(['Date', 'workoutActivityType']) # sort by date and workout activity type

# Pivot to find out type of workouts done that day
# wm = lambda x: np.average(x, weights = final_workout_df.loc[x.index, 'duration']) # lambda function to compute weighted mean with duration as weights
final_workout_df_gp = final_workout_df.groupby(['workoutActivityType', 'Date', 'Day'], as_index=False).agg({'duration':'sum'})
                      # daily workout aggregation (sum for duration)
final_workout_df_daily = final_workout_df_gp.drop(columns=['workoutActivityType', 'duration']).join(pd.get_dummies(final_workout_df_gp['workoutActivityType']))
                         # use get_dummies to get the type of workout done for each day
final_workout_df_daily = final_workout_df_daily.groupby(['Date', 'Day'], as_index=False).max()

# Add the type of workout on to daily_pivot
start_date = final_workout_df_daily['Date'].min() # start date is determined by workout minimum data at 2023-03-21
record_data_pivot = record_data_pivot[record_data_pivot['Date'] >= start_date]

final_df = pd.merge(record_data_pivot, final_workout_df_daily, how='left', on=['Date', 'Day']) # adding workout data onto record_data_pivot


'''
# Plotting
# Basal Energy Burned while using Apple Watch
start_date = '2023-03-19' # start date of wearing Apple Watch


basal_energy_daily = record_data_daily[record_data_daily['type'] == 'ActiveEnergyBurned' & record_data_daily['Date'] >= start_date]
fig = px.line(basal_energy_daily, x=basal_energy_daily['Date'], y=basal_energy_daily['value'], markers=True)
fig.update_layout(title_text="Basal Energy Progress Since Wearing Apple Watch")
fig.show()
'''
