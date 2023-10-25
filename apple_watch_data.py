# Import packages
import pandas as pd
import xml.etree.ElementTree as ET
import io

path = '/Users/simrannayak/Desktop/apple_health_export/export.xml' # load data

# Create element tree object
tree = ET.parse(path)
root = tree.getroot()

# Health Record Data
record_list = [x.attrib for x in root.iter('Record')] # for every health record, extract attributes
record_data = pd.DataFrame(record_list) # change to dataframe
record_data['duration'] = (pd.to_datetime(record_data['endDate']) - pd.to_datetime(record_data['startDate'])).astype('timedelta64[D]')

record_data = record_data.drop(['sourceName', 'sourceVersion', 'device', 'creationDate', 'endDate'], axis=1) # drop unnecessary columns

record_data['Day'] = pd.to_datetime(record_data['startDate']).dt.strftime('%A') # extract day of startdate
record_data['Date'] = pd.to_datetime(record_data['startDate']).dt.strftime('%Y-%m-%d') # extract date of startdate
record_data['Month'] = pd.to_datetime(record_data['startDate']).dt.strftime('%B') # extract month of startdate

record_data['value'] = pd.to_numeric(record_data['value'], errors='coerce') # change to numeric
record_data['value'] = record_data['value'].fillna(1.0) # if NaN then change to occurence (1.0)
record_data['type'] = record_data['type'].str.replace('HKQuantityTypeIdentifier', '') # shorten observation name
record_data['type'] = record_data['type'].str.replace('HKCategoryTypeIdentifier', '')

record_data = record_data[['type', 'Date', 'Day', 'Month', 'value', 'unit']] # reorder columns
record_data = record_data.sort_values(['Date', 'type']) # sort by date and type of record


# Daily Health Record
daily_record_list = ['StepCount', 'DistanceWalkingRunning', 'BasalEnergyBurned',
                     'ActiveEnergyBurned', 'FlightsClimbed', 'AppleExerciseTime',
                     'DistanceCycling', 'DistanceSwimming', 'AppleStandTime', 'AppleStandHour'] 
                     # selecting columns to create daily record aggregated by sum

record_data_daily = record_data[record_data['type'].isin(daily_record_list)]
record_data_daily = record_data_daily.groupby(['type', 'Date', 'Day', 'Month', 'unit'], as_index=False).agg({'value': 'sum'})


# Workout Data
final_workout_dict = [] # create dictionary to add workout statistics in
workout_list = list(root.iter('Workout')) # grab workout data

for i in range(len(workout_list)): # iterating through each workout
    workout_dict = workout_list[i].attrib 
    average_list = ['HeartRate', 'RunningSpeed', 'RunningPower', 'RunningGroundContactTime',
                    'RunningVerticalOscillation', 'RunningStrideLength'] # workout metrics that have an 'average' value
    
    WorkoutStatisticsList = list(workout_list[i].iter('WorkoutStatistics')) # grab statistics for each workout
    for i, WorkoutStatistics in enumerate(WorkoutStatisticsList): # iterating through workout statistics
        type = WorkoutStatistics.attrib['type'].replace('HKQuantityTypeIdentifier', '') # clean up workout statistics name

        # workout metrics in WorkoutStatistics are two types: those that have an 'average' data value and a 'sum' value
        # the following if-else statement uses average_list to determine if we need to grab 'average' or 'sum'
        if type in average_list:
            workout_dict[type] = WorkoutStatistics.attrib['average']
        else:
            workout_dict[type] = WorkoutStatistics.attrib['sum']
    
    final_workout_dict.append(workout_dict) # appending workout statistics as columns for each workout 

final_workout_df = pd.DataFrame(final_workout_dict) # create final_workout_df dataframe

final_workout_df = final_workout_df.drop(['sourceName','sourceVersion', 'device', 'creationDate','endDate'], axis=1) # remove unnecessary columns

final_workout_df['Day'] = pd.to_datetime(final_workout_df['startDate']).dt.strftime('%A') # extract day of startdate
final_workout_df['Date'] = pd.to_datetime(final_workout_df['startDate']).dt.strftime('%Y-%m-%d') # extract date of startdate
final_workout_df['Month'] = pd.to_datetime(final_workout_df['startDate']).dt.strftime('%B') # extract month of startdate

final_workout_df['workoutActivityType'] = final_workout_df['workoutActivityType'].str.replace('HKWorkoutActivityType', '') # clean up activity name
final_workout_df['duration'] = final_workout_df['duration'].astype(float) # transform duration into float
final_workout_df['ActiveEnergyBurned'] = final_workout_df['ActiveEnergyBurned'].astype(float) # transform energy burnt into float
final_workout_df['BasalEnergyBurned'] = final_workout_df['BasalEnergyBurned'].astype(float) # transform energy burnt into float

final_workout_df = final_workout_df[['workoutActivityType', 'Date', 'Day', 'Month', 'duration',
                                     'durationUnit', 'BasalEnergyBurned', 'ActiveEnergyBurned', 
                                     'HeartRate', 'DistanceWalkingRunning']] # reorder columns and only use key columns
                                     # columns not used: StepCount, RunningSpeed, RunningPower, RunningGroundContactTime,
                                     # RunningGroundContactTime, RunningVerticalOscillation, RunningStrideLength, DistanceSwimming, 
                                     # SwimmingStrokeCount, DistanceCycling
final_workout_df = final_workout_df.sort_values(['Date', 'workoutActivityType']) # sort by date and workout activity type