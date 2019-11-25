import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from scipy import stats
from scipy.stats import norm
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, recall_score, accuracy_score, precision_score
from sklearn.metrics import precision_recall_curve
from deap import algorithms
from deap import base
from deap import creator
from deap import tools
import random
import pickle
import timeit
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
pd.options.mode.chained_assignment = None  # default='warn'
start_time = timeit.default_timer()

# _______________READ AND FORMAT FILES____________________
telemetry = pd.read_csv('PdM_telemetry.csv')
errors = pd.read_csv('PdM_errors.csv')
maint = pd.read_csv('PdM_maint.csv')
failures = pd.read_csv('PdM_failures.csv')
machines = pd.read_csv('PdM_machines.csv')

# format datetime field which comes in as string
telemetry['datetime'] = pd.to_datetime(telemetry['datetime'], format="%Y-%m-%d %H:%M:%S")

# format datetime field which comes in as string
errors['datetime'] = pd.to_datetime(errors['datetime'], format="%Y-%m-%d %H:%M:%S")
errors['errorID'] = errors['errorID'].astype('category')

# format datetime field which comes in as string
maint['datetime'] = pd.to_datetime(maint['datetime'], format="%Y-%m-%d %H:%M:%S")
maint['comp'] = maint['comp'].astype('category')

machines['model'] = machines['model'].astype('category')
#print(machines['model'].iloc[-1]) # last row of the dataframe

# format datetime field which comes in as string
failures['datetime'] = pd.to_datetime(failures['datetime'], format="%Y-%m-%d %H:%M:%S")

# ___________________LAG FEATURES_________________________
# Calculate 3H mean values for telemetry features
temp = []
fields = ['volt', 'rotate', 'pressure', 'vibration']
for col in fields:
    temp.append(pd.pivot_table(telemetry,
                               index='datetime',
                               columns='machineID',
                               values=col).resample('3H', closed='left', label='right').mean().unstack())
telemetry_mean_3h = pd.concat(temp, axis=1)
telemetry_mean_3h.columns = [i + 'mean_3h' for i in fields]
telemetry_mean_3h.reset_index(inplace=True)
# Calculate 3H standard deviation for telemetry features
temp = []
for col in fields:
    temp.append(pd.pivot_table(telemetry,
                               index='datetime',
                               columns='machineID',
                               values=col).resample('3H', closed='left', label='right').std().unstack())
telemetry_sd_3h = pd.concat(temp, axis=1)
telemetry_sd_3h.columns = [i + 'sd_3h' for i in fields]
telemetry_sd_3h.reset_index(inplace=True)

# Calculate 24H mean values for telemetry features
temp = []
fields = ['volt', 'rotate', 'pressure', 'vibration']
for col in fields:
    temp.append(pd.rolling_mean(pd.pivot_table(telemetry,
                                               index='datetime',
                                               columns='machineID',
                                               values=col), window=24).resample('3H',
                                                                                closed='left',
                                                                                label='right').first().unstack())
telemetry_mean_24h = pd.concat(temp, axis=1)
telemetry_mean_24h.columns = [i + 'mean_24h' for i in fields]
telemetry_mean_24h.reset_index(inplace=True)
telemetry_mean_24h = telemetry_mean_24h.loc[-telemetry_mean_24h['voltmean_24h'].isnull()]
# Calculate 24H standard deviation for telemetry features
temp = []
fields = ['volt', 'rotate', 'pressure', 'vibration']
for col in fields:
    temp.append(pd.rolling_std(pd.pivot_table(telemetry,
                                              index='datetime',
                                              columns='machineID',
                                              values=col), window=24).resample('3H',
                                                                               closed='left',
                                                                               label='right'
                                                                               ).first().unstack())
telemetry_sd_24h = pd.concat(temp, axis=1)
telemetry_sd_24h.columns = [i + 'sd_24h' for i in fields]
telemetry_sd_24h.reset_index(inplace=True)
telemetry_sd_24h = telemetry_sd_24h.loc[-telemetry_sd_24h['voltsd_24h'].isnull()]
# Notice that a 24h rolling average is not available at the earliest timepoints

# Merge datasets - 3H and 24H
telemetry_feat = pd.concat([telemetry_mean_3h,
                            telemetry_sd_3h.ix[:, 2:6],
                            telemetry_mean_24h.ix[:, 2:6],
                            telemetry_sd_24h.ix[:, 2:6]], axis=1).dropna()
telemetry_feat.describe()

# Lag Features from errors
error_count = pd.get_dummies(errors.set_index('datetime')).reset_index()  # create a column for each error type
error_count.columns = ['datetime', 'machineID', 'error1', 'error2', 'error3', 'error4', 'error5']
# combine errors for a given machine in a given hour -> dataset only with errors
error_count = error_count.groupby(['machineID', 'datetime']).sum().reset_index()
# add the dataset with all the hours without errors
error_count = telemetry[['datetime', 'machineID']].merge(error_count, on=['machineID', 'datetime'], how='left').fillna(0.0)

# error count for the last 24h
temp = []
fields = ['error%d' % i for i in range(1, 6)]
for col in fields:
    temp.append(pd.rolling_sum(pd.pivot_table(error_count,
                                              index='datetime',
                                              columns='machineID',
                                              values=col), window=24).resample('3H',
                                                                               closed='left',
                                                                               label='right'
                                                                               ).first().unstack())
error_count = pd.concat(temp, axis=1)
error_count.columns = [i + 'count' for i in fields]
error_count.reset_index(inplace=True)
error_count = error_count.dropna()
error_count.describe()

# Days Since Last Replacement from Maintenance
comp_rep = pd.get_dummies(maint.set_index('datetime')).reset_index()  # create a column for each error type
comp_rep.columns = ['datetime', 'machineID', 'comp1', 'comp2', 'comp3', 'comp4']

# combine repairs for a given machine in a given hour
comp_rep = comp_rep.groupby(['machineID', 'datetime']).sum().reset_index()

# add timepoints where no components were replaced
comp_rep = telemetry[['datetime', 'machineID']].merge(comp_rep,
                                                      on=['datetime', 'machineID'],
                                                      how='outer').fillna(0).sort_values(by=['machineID', 'datetime'])

components = ['comp1', 'comp2', 'comp3', 'comp4']
for comp in components:
    # convert indicator to most recent date of component change
    comp_rep.loc[comp_rep[comp] < 1, comp] = None
    comp_rep.loc[-comp_rep[comp].isnull(), comp] = comp_rep.loc[-comp_rep[comp].isnull(), 'datetime']
    # forward-fill the most-recent date of component change
    comp_rep[comp] = comp_rep[comp].fillna(method='ffill')

# remove dates in 2014 (may have NaN or future component change dates)
comp_rep = comp_rep.loc[comp_rep['datetime'] > pd.to_datetime('2015-01-01')]

# replace dates of most recent component change with days since most recent component change
for comp in components:
    comp_rep[comp] = (comp_rep['datetime'] - pd.to_datetime(comp_rep[comp])) / np.timedelta64(1, 'D')

comp_rep.describe()

# Final Dataset
final_feat = telemetry_feat.merge(error_count, on=['datetime', 'machineID'], how='left')
final_feat = final_feat.merge(comp_rep, on=['datetime', 'machineID'], how='left')
final_feat = final_feat.merge(machines, on=['machineID'], how='left')

# __________________LABEL CREATION_____________________

labeled_features = final_feat.merge(failures, on=['datetime', 'machineID'], how='left')
delta = 7 # 7x3=21+3=24: comp4 error for 3h measure - fills error comp4 for the previous 21h
components = ['comp1', 'comp2', 'comp3', 'comp4']

for comp in components:
    df = labeled_features.index[labeled_features['failure'] == comp].tolist()
    for ind1 in df:
        labeled_features.ix[range(ind1 - delta, ind1, 1), 'failure'] = comp

labeled_features.ix[labeled_features.index[pd.isnull(labeled_features['failure'])].tolist(), 'failure'] = 'none'
labeled_features['failure'].astype('category')

# __________________CLASSIFIER TRAINING _________________

# threshold_dates = [[pd.to_datetime('2015-05-31 01:00:00'), pd.to_datetime('2015-06-01 01:00:00')],
#                    [pd.to_datetime('2015-06-30 01:00:00'), pd.to_datetime('2015-07-01 01:00:00')],
#                    [pd.to_datetime('2015-07-31 01:00:00'), pd.to_datetime('2015-08-01 01:00:00')],
#                    [pd.to_datetime('2015-08-31 01:00:00'), pd.to_datetime('2015-09-01 01:00:00')]]
#
# test_results = []
# models = []
# for last_train_date, first_test_date in threshold_dates:
#
#     print('Training a model: Last date to train: ' , last_train_date)
#     last_test_date = pd.to_datetime('2015-10-01 01:00:00')
#     # split out training and test data
#     train_y = labeled_features.loc[labeled_features['datetime'] < last_train_date, 'failure']
#     train_X = pd.get_dummies(labeled_features.loc[labeled_features['datetime'] < last_train_date].drop(['datetime',
#                                                                                                         'machineID',
#                                                                                                         'failure'], 1))
#     test_X = pd.get_dummies(labeled_features.loc[(labeled_features['datetime'] > first_test_date) & (labeled_features['datetime'] < last_test_date)].drop(['datetime',
#                                                                                                        'machineID',
#                                                                                                        'failure'], 1))
#
#     # train and predict using the model, storing results for later
#     my_model = GradientBoostingClassifier(random_state=42)
#     my_model.fit(train_X, train_y)
#     test_result = pd.DataFrame(labeled_features.loc[(labeled_features['datetime'] > first_test_date) & (labeled_features['datetime'] < last_test_date)])
#     test_result['predicted_failure'] = my_model.predict(test_X)
#     test_results.append(test_result)
#     models.append(my_model)
#
#     print('Model training complete...')

# _________________CLASSIFIER EVALUATION______________________

# def Evaluate(predicted, actual, labels):
#     output_labels = []
#     output = []
#
#     # Calculate and display confusion matrix
#     cm = confusion_matrix(actual, predicted, labels=labels)
#     print('Confusion matrix\n- x-axis is true labels (none, comp1, etc.)\n- y-axis is predicted labels')
#     print(cm)
#
#     # Calculate precision, recall, and F1 score
#     accuracy = np.array([float(np.trace(cm)) / np.sum(cm)] * len(labels))
#     precision = precision_score(actual, predicted, average=None, labels=labels)
#     recall = recall_score(actual, predicted, average=None, labels=labels)
#     f1 = 2 * precision * recall / (precision + recall)
#     output.extend([accuracy.tolist(), precision.tolist(), recall.tolist(), f1.tolist()])
#     output_labels.extend(['accuracy', 'precision', 'recall', 'F1'])
#
#     # Calculate the macro versions of these metrics
#     output.extend([[np.mean(precision)] * len(labels),
#                    [np.mean(recall)] * len(labels),
#                    [np.mean(f1)] * len(labels)])
#     output_labels.extend(['macro precision', 'macro recall', 'macro F1'])
#
#     # Find the one-vs.-all confusion matrix
#     cm_row_sums = cm.sum(axis=1)
#     cm_col_sums = cm.sum(axis=0)
#     s = np.zeros((2, 2))
#     for i in range(len(labels)):
#         v = np.array([[cm[i, i],
#                        cm_row_sums[i] - cm[i, i]],
#                       [cm_col_sums[i] - cm[i, i],
#                        np.sum(cm) + cm[i, i] - (cm_row_sums[i] + cm_col_sums[i])]])
#         s += v
#     s_row_sums = s.sum(axis=1)
#
#     # Add average accuracy and micro-averaged  precision/recall/F1
#     avg_accuracy = [np.trace(s) / np.sum(s)] * len(labels)
#     micro_prf = [float(s[0, 0]) / s_row_sums[0]] * len(labels)
#     output.extend([avg_accuracy, micro_prf])
#     output_labels.extend(['average accuracy',
#                           'micro-averaged precision/recall/F1'])
#
#     # Compute metrics for the majority classifier
#     mc_index = np.where(cm_row_sums == np.max(cm_row_sums))[0][0]
#     cm_row_dist = cm_row_sums / float(np.sum(cm))
#     mc_accuracy = 0 * cm_row_dist;
#     mc_accuracy[mc_index] = cm_row_dist[mc_index]
#     mc_recall = 0 * cm_row_dist;
#     mc_recall[mc_index] = 1
#     mc_precision = 0 * cm_row_dist
#     mc_precision[mc_index] = cm_row_dist[mc_index]
#     mc_F1 = 0 * cm_row_dist;
#     mc_F1[mc_index] = 2 * mc_precision[mc_index] / (mc_precision[mc_index] + 1)
#     output.extend([mc_accuracy.tolist(), mc_recall.tolist(),
#                    mc_precision.tolist(), mc_F1.tolist()])
#     output_labels.extend(['majority class accuracy', 'majority class recall',
#                           'majority class precision', 'majority class F1'])
#
#     # Random accuracy and kappa
#     cm_col_dist = cm_col_sums / float(np.sum(cm))
#     exp_accuracy = np.array([np.sum(cm_row_dist * cm_col_dist)] * len(labels))
#     kappa = (accuracy - exp_accuracy) / (1 - exp_accuracy)
#     output.extend([exp_accuracy.tolist(), kappa.tolist()])
#     output_labels.extend(['expected accuracy', 'kappa'])
#
#     # Random guess
#     rg_accuracy = np.ones(len(labels)) / float(len(labels))
#     rg_precision = cm_row_dist
#     rg_recall = np.ones(len(labels)) / float(len(labels))
#     rg_F1 = 2 * cm_row_dist / (len(labels) * cm_row_dist + 1)
#     output.extend([rg_accuracy.tolist(), rg_precision.tolist(),
#                    rg_recall.tolist(), rg_F1.tolist()])
#     output_labels.extend(['random guess accuracy', 'random guess precision',
#                           'random guess recall', 'random guess F1'])
#
#     # Random weighted guess
#     rwg_accuracy = np.ones(len(labels)) * sum(cm_row_dist ** 2)
#     rwg_precision = cm_row_dist
#     rwg_recall = cm_row_dist
#     rwg_F1 = cm_row_dist
#     output.extend([rwg_accuracy.tolist(), rwg_precision.tolist(),
#                    rwg_recall.tolist(), rwg_F1.tolist()])
#     output_labels.extend(['random weighted guess accuracy',
#                           'random weighted guess precision',
#                           'random weighted guess recall',
#                           'random weighted guess F1'])
#
#     output_df = pd.DataFrame(output, columns=labels)
#     output_df.index = output_labels
#
#     return output_df
#
# evaluation_results = []
# for i, test_result in enumerate(test_results):
#     print('\nSplit %d:' % (i+1))
#     evaluation_result = Evaluate(actual = test_result['failure'],
#                                  predicted = test_result['predicted_failure'],
#                                  labels = ['none', 'comp1', 'comp2', 'comp3', 'comp4'])
#     evaluation_results.append(evaluation_result)

#last_train_date = pd.to_datetime('2015-10-01 01:00:00')
last_train_date = pd.to_datetime('2015-10-01 01:00:00')
# print('Training model with 9 months')
# # split out training and test data
train_y = labeled_features.loc[labeled_features['datetime'] < last_train_date, 'failure']
train_X = pd.get_dummies(labeled_features.loc[labeled_features['datetime'] < last_train_date].drop(['datetime',
                                                                                                    'machineID',
                                                                                                     'failure'], 1))
# # train model
# my_model = GradientBoostingClassifier(random_state=42)
# my_model.fit(train_X, train_y) # fit model on 9 months of data
# print('Model training complete')

# # save the model to disk
filename = 'finalized_model.sav'
# pickle.dump(my_model, open(filename, 'wb'))
# print('model saved')

# load the model from disk
#loaded_model = pickle.load(open(filename, 'rb'))


# _________________ GRAPHS_________________

#cena = pd.to_datetime(X.loc[(X['machineID'] == 4) & (X['failure'] == 'comp2')& (X['datetime'] > pd.to_datetime('2015-10-01 01:00:00')), ['datetime']].values[7])
#cena2 = pd.to_datetime(X.loc[(X['machineID'] == 3) & (X['failure'] == 'comp2')& (X['datetime'] > pd.to_datetime('2015-10-01 01:00:00')), ['datetime']].values[14])

# plot_df = X.loc[(X['machineID'] == 4) &
#                 (X['datetime'] > pd.to_datetime('2015-10-01 01:00:00')), ['datetime', 'pressuremean_3h', 'voltmean_3h', 'vibrationmean_3h', 'rotatemean_3h']]
# sns.set_style("darkgrid")
# plt.figure(figsize=(12, 6))
# plt.plot(plot_df['datetime'], plot_df['pressuremean_3h'], 'r', label= 'pressuremean_3h')
# plt.plot(plot_df['datetime'], plot_df['voltmean_3h'], 'b', label = 'voltmean_3h')
# plt.plot(plot_df['datetime'], plot_df['vibrationmean_3h'], 'g', label = 'vibrationmean_3h')
# plt.plot(plot_df['datetime'], plot_df['rotatemean_3h'], 'y', label='rotatemean_3h')
# plt.ylabel('Sensors 3H mean')
# plt.axvline(x=cena[0], color='c', label = 'comp2 failure')
# #plt.axvline(x=cena2[0], color='c')
# plt.legend()
#
# # make x-axis ticks legible
# adf = plt.gca().get_xaxis().get_major_formatter()
# adf.scaled[1.0] = '%m-%d'
# plt.xlabel('Date')
# plt.show()

# _________________ SIMULATION ____________________

simul_begin = pd.to_datetime('2015-11-01 00:00:00') # datetime of simulation beginning
simul_end = pd.to_datetime('2016-12-01 00:00:00')   # datetime to simulation ending
simul_dataframe = (labeled_features.loc[labeled_features['datetime'] <= simul_end])
simul_dataframe = simul_dataframe.loc[simul_dataframe['datetime'] >= simul_begin]
X = pd.DataFrame([]) # data frame where simulation data is appended

total_cycles = 448# 448 = 56*8blocks of 3H - 56 days - 8 weeks
current_cycle = 0
prediction = ['none', 'none', 'none']
id_failed = 0 # stores the ID of the machine that failed when GA called
week_comp = 0 # to check if a week has passed (when 56 = week complete)
week_count = 0 # number of weeks passed
in_ga = 0 # 0 if not in genetic algorithm, 1 if in genetic algorithm
start_ga = 0 #flag to start genetic algorithm, 1 if suposed to start
prod = 0 # production each iteration
total_mach= 3  # number of machines
load = np.array([None, 0, 0, 0, 0, 0]) # current load for each machine ID
flag_state = np.array([None, 0, 0, 0, 0, 0]) # 0 if real data normal, 1 if maintenance per machine ID, 2 if simulated normal, 3 if simulated failure
return_fmaint = np.array([None, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]) # when 1 means it returned from maintenance mode. Per machine ID
in_maint = np.array([None, 0, 0, 0, 0, 0]) # cycles of 3H blocks passed in maintenance per machine ID
in_normal = np.array([None, 0, 0, 0, 0, 0]) # hours passed in normal generated data per machine ID (adds 3h each cycle passed)
in_fail = np.array([None, 0, 0, 0, 0, 0])  # cycles passed in failure generated data per machine ID
mttr = np.array([None, 9, 9, 9, 9, 9]) # mean time to repair per machine ID (defined)
mtbf = np.array([[None, 4320 * 1.1, 3408 * 1.1, 1800 * 1.1, 2160 * 1.1, 3240 * 1.1, 2640 * 1.2, 1080 * 1.2,
                  1080 * 1.2,  # load -2
                  1200 * 1.2, 2160 * 1.2, 3240 * 1.2, 1980 * 1.2, 4320 * 1.2, 4317 * 1.2, 3418 * 1.2, 900 * 1.2],
                 [None, 4320 * 1.05, 3408 * 1.05, 1800 * 1.05, 2160 * 1.05, 3240 * 1.05, 2640 * 1.1, 1080 * 1.1,
                  1080 * 1.1,
                  1200 * 1.1, 2160 * 1.1, 3240 * 1.1, 1980 * 1.1, 4320 * 1.1, 4317 * 1.1, 3418 * 1.1, 900 * 1.1],
                 # load -1
                 [None, 4320, 3408, 1800, 2160, 3240, 2640, 1080, 1080, 1200, 160, 3240, 1980, 4320, 4317, 3418,
                  900],  # load 0
                 [None, 4320 / 1.05, 3408 / 1.05, 1800 / 1.05, 2160 / 1.05, 3240 / 1.05, 2640 / 1.1, 1080 / 1.1,
                  1080 / 1.5, 1200 / 5,
                  2160 / 5, 3240 / 5, 1980 / 5, 4320 / 5, 4317 / 5, 3418 / 5, 900 / 5],  # load +1
                 [None, 4320 / 1.1, 3408 / 1.1, 1800 / 1.1, 2160 / 1.1, 3240 / 1.1, 2640 / 2, 1080 / 2, 1080 / 2,
                  1200 / 2,
                  2160 / 10, 3240 / 10, 1980 / 10,
                  4320 / 10, 4317 / 10, 3418 / 10, 900 / 10],  # load +2
                 ]) # meantime between failure calculated per machine ID: mtbf[x][ID] - x = load +2 => ex: x= 1 load-1...
index = np.array([0, 0, 0, 0, 0, 0]) # index for current simulation data frame window for each machine ID and total (sum for every ID) in [0]
current_simul_date = np.array([None, simul_begin, simul_begin, simul_begin, simul_begin]) # current simulation timestamp for each machine ID
do_maint = np.array([None, 0, 0, 0, 0, 0, 0, 0]) # shows if maintenance is being performed or not : 1 - enters maintenance mode; 0 - not
production = np.array([None, 15, 6.87, 21, 15])# Production rate per machine ID (defined)
target = np.array([2400, 1300, 1200, 1000]) # Weekly target (defined)
maint_df = pd.DataFrame(columns=['datetime', 'machineID', 'comp']) # empty dataframe for maintenance scheduling;
maint_df['datetime'] = pd.to_datetime(maint_df['datetime'], format="%Y-%m-%d %H:%M:%S")
maint_df['comp'] = maint_df['comp'].astype('category')
recent_maint_matrix = np.full((4, 5), 0, dtype=object) # matrix with maintenance not performed: columns: machine ID, lines: components

mean = np.zeros(shape=(total_mach, 16))  # mean value of normal dist fit: rows : machine ID, columns: mean for each sensor features (volt3hmean, volt24hmean...)
std = np.zeros(shape=(total_mach, 16))   # standard deviation of normal dist fit
mean_fail = np.zeros(shape=(total_mach, 16)) # mean value of normal dist fit in 24h previous to failure
std_fail = np.zeros(shape=(total_mach, 16))  # std of normal dist fir in 24h previous to failure

# ___________________ Normal distribution________________
# # Calculate mean and std parameters for normal distribution fit to each sensor feature for each machine
# No failure data from training set
train_set = labeled_features.loc[labeled_features['datetime'] < simul_begin]
no_failure_train = train_set.loc[train_set['failure'] == 'none']
for ID1 in range(1, total_mach + 1):
    for cols in range(0, 16):
        no_failure = no_failure_train.loc[no_failure_train['machineID'] == ID1]
        no_failure = no_failure.iloc[:, 2:18]  # columns of sensor values
        b = no_failure.iloc[:, cols]
        mu, std1 = norm.fit(b)  # fit normal distribution to values
        mean[(ID1 - 1), cols] = mu
        std[(ID1 - 1), cols] = std1

# comp2 failure data
failure_train = train_set.loc[train_set['failure'] == 'comp2']
for ID3 in range(1, total_mach + 1):
    for cols in range(0, 16):
        failure = failure_train.loc[failure_train['machineID'] == ID3]
        failure = failure.iloc[:, 2:18]  # columns of sensor values
        bb = failure.iloc[:, cols]
        mu2, std2 = norm.fit(bb)  # fit normal distribution to values
        mean_fail[(ID3 - 1), cols] = mu2
        std_fail[(ID3 - 1), cols] = std2

print(mean)
print(mean_fail)
print(std)
print(std_fail)
# ____________________MTBF_____________________________
# # Calculates MTBF in load0 for each machine ID based on real data
# mtbf=[]
# h1=[]
# x_l1 = labeled_features.loc[labeled_features['failure'] == 'comp2']
# # calculate mtbf from real data for each machine ID
# for o in range(1,101):
#     tdelta1 = []
#     x_l2 = x_l1.loc[x_l1['machineID'] == o]
#     aa1 = len(x_l2)
#     if (aa1 > 9):
#         for i in range(7, aa1-8, 8):
#             diff1 = x_l2.datetime.iloc[i] - x_l2.datetime.iloc[i+8]
#             days1, seconds1 = diff1.days, diff1.seconds
#             hours1 = days1 *24 + seconds1 // 3600
#             tdelta1.append(abs(hours1))
#     elif aa1 == 9:
#         diff1 = x_l2.datetime.iloc[0] - x_l2.datetime.iloc[8]
#         days1, seconds1 = diff1.days, diff1.seconds
#         hours1 = days1 * 24 + seconds1 // 3600
#         tdelta1.append(abs(hours1))
#         tdelta1.append(abs(hours1))
#     else:
#         tdelta1=0
#     if aa1>=14:
#         h1=np.mean(tdelta1)
#     elif aa1>7 & aa1<14:
#         h1=np.mean(tdelta1)
#     else:
#         h1=0
#     mtbf.append(h1)   # mean time between failures per machineID mtbf[0] = mtbf of ID1, mtbf[1] = mtbf of ID2...


# _____________ Defining Functions _____________________
#
def main_GA():
    random.seed(64)
    NGEN = 100 #generations
    MU = 80 #individuals
    LAMBDA = 200 #number of children each generation
    CXPB = 0.7 #crossover rate
    MUTPB = 0.25#mutation rate

    pop = toolbox.population(n=MU)
    hof = tools.HallOfFame(1) #contain an infinity of different individuals
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    algorithms.eaMuPlusLambda(pop, toolbox, MU, LAMBDA, CXPB, MUTPB, NGEN, stats,
                              halloffame=hof) #algorithm takes in a population and evolves it

    return pop, stats, hof

def evalKnapsack(individual):
    print(individual)
    with open('save.pkl', 'rb') as f :
        w_res1= pickle.load(f)
        w_quo1= pickle.load(f)
        current_simul_date1= pickle.load(f)
        prod1= pickle.load(f)
        id_failed1 = pickle.load(f)
    f.close()
# fixed variables:
    scale_load_dec = np.array([1.1, 1.05, 1, 0.9, 0.8])  # scale - load -2 to +2 when load decreases
    scale_load_inc = np.array([0.8, 0.9, 1, 1.05, 1.1])  # scale - load -2 to +2 when load increases
    total_mach = 3
    mttr = np.array([None, 9, 9, 9, 9, 9])  # mean time to repair per machine ID (defined)
    mtbf = np.array([[None, 4320 * 1.1, 3408 * 1.1, 1800 * 1.1, 2160 * 1.1, 3240 * 1.1, 2640 * 1.2, 1080 * 1.2,
                      1080 * 1.2,  # load -2
                      1200 * 1.2, 2160 * 1.2, 3240 * 1.2, 1980 * 1.2, 4320 * 1.2, 4317 * 1.2, 3418 * 1.2, 900 * 1.2],
                     [None, 4320 * 1.05, 3408 * 1.05, 1800 * 1.05, 2160 * 1.05, 3240 * 1.05, 2640 * 1.1, 1080 * 1.1,
                      1080 * 1.1,
                      1200 * 1.1, 2160 * 1.1, 3240 * 1.1, 1980 * 1.1, 4320 * 1.1, 4317 * 1.1, 3418 * 1.1, 900 * 1.1],
                     # load -1
                     [None, 4320, 3408, 1800, 2160, 3240, 2640, 1080, 1080, 1200, 160, 3240, 1980, 4320, 4317, 3418,
                      900],  # load 0
                     [None, 4320 / 1.05, 3408 / 1.05, 1800 / 1.05, 2160 / 1.05, 3240 / 1.05, 2640 / 1.1, 1080 / 1.1,
                      1080 / 1.5, 1200 / 5,
                      2160 / 5, 3240 / 5, 1980 / 5, 4320 / 5, 4317 / 5, 3418 / 5, 900 / 5],  # load +1
                     [None, 4320 / 1.1, 3408 / 1.1, 1800 / 1.1, 2160 / 1.1, 3240 / 1.1, 2640 / 2, 1080 / 2, 1080 / 2, 1200 / 2,
                      2160 / 10, 3240 / 10, 1980 / 10,
                      4320 / 10, 4317 / 10, 3418 / 10, 900 / 10],  # load +2
                     ])
    day = [np.array([]), np.array([]), np.array([])] # change if more machines
    for ID in range(1, total_mach + 1):
        day[ID - 1] = (
        [pd.to_datetime(current_simul_date1[ID]), pd.to_datetime(current_simul_date1[ID] + pd.to_timedelta(1, unit='D')),
         pd.to_datetime(current_simul_date1[ID] + pd.to_timedelta(2, unit='D')),
         pd.to_datetime(current_simul_date1[ID] + pd.to_timedelta(3, unit='D')),
         pd.to_datetime(current_simul_date1[ID] + pd.to_timedelta(4, unit='D')),
         pd.to_datetime(current_simul_date1[ID] + pd.to_timedelta(5, unit='D')),
         pd.to_datetime(current_simul_date1[ID] + pd.to_timedelta(6, unit='D')),
         pd.to_datetime(current_simul_date1[ID] + pd.to_timedelta(7, unit='D')),
         pd.to_datetime(current_simul_date1[ID] + pd.to_timedelta(7, unit='D'))])
    day = np.vstack(day)
    target = np.array([2400, 1300, 1200, 1000])  # Weekly target (defined)
    simul_begin = pd.to_datetime('2015-10-01 00:00:00') # datetime of simulation beginning
    production = np.array([None, 5, 2.29, 7, 2])# Production rate per machine ID (defined) per hour

# change each call
    next_fail_time = np.array([pd.to_datetime(0), pd.to_datetime(0), pd.to_datetime(0),
                               pd.to_datetime(0), pd.to_datetime(0),pd.to_datetime(0)]) #next fail timestamp per machine ID-1
    maintenance_time = np.array([pd.to_datetime(0), pd.to_datetime(0), pd.to_datetime(0),
                               pd.to_datetime(0), pd.to_datetime(0),pd.to_datetime(0)]) #start maintenace date per machine ID-1
    fail_in = np.array([0, 0, 0, 0, 0, 0]) # nr of fails per machine ID-1 in week
    fail_out = 0 #nr of fails next week
    size = w_quo1
    in_week_last = current_simul_date1[1] + pd.to_timedelta(size, unit='D')  # last date of the week
    next_week_last = current_simul_date1[1] + pd.to_timedelta(size, unit='D')+ pd.to_timedelta(2, unit='D') # last date of the 2 days of next week

# Calculates fail_in (nr of fails in current week) and fail_out (nr fails next week):
    b = np.array([pd.to_timedelta(0), pd.to_timedelta(0), pd.to_timedelta(0), pd.to_timedelta(0), pd.to_timedelta(0)])
    b2 = np.array([pd.to_timedelta(0), pd.to_timedelta(0), pd.to_timedelta(0), pd.to_timedelta(0), pd.to_timedelta(0)])
    a = np.array([pd.to_datetime(0), pd.to_datetime(0), pd.to_datetime(0), pd.to_datetime(0), pd.to_datetime(0),
                  pd.to_datetime(0)])

    for mach in range(0, total_mach):
        for indiv in range((mach * size) + 1, ((size - 1) + (mach * size) + 1)):
            if ((individual[((mach * size))]) - individual[indiv]) < 0:
                b[mach] = -pd.to_timedelta(24, unit='H') * scale_load_dec[individual[indiv] + 2]
            elif ((individual[((mach * size))]) - individual[indiv]) > 0:
                b[mach] = +pd.to_timedelta(24, unit='H') * scale_load_inc[individual[indiv] + 2]
            else:
                b[mach] = pd.to_timedelta(0, unit='H')
            b2[mach] = b2[mach] + b[mach]

    for mach in range(0, total_mach):
        if mach == (id_failed1 - 1):
            a[mach] = current_simul_date1[id_failed1] + pd.to_timedelta(1, unit='D') - pd.to_timedelta(3, unit='H')
            next_fail_time[mach] = a[mach]
        else:
            a[mach] = simul_begin + pd.to_timedelta(mtbf[(individual[(size - 1) + (size * mach)])+2][mach + 1], unit='H')
            next_fail_time[mach] = a[mach] + b2[mach]

# Calculates maintenance time
    for ID in range(0, total_mach):
        if next_fail_time[ID] >= day[ID][0]:
            if next_fail_time[ID].time().strftime('%H:%M:%S') < '12:00:00':
                maintenance_time[ID] = next_fail_time[ID].replace(hour=6, minute=0)
            else:
                maintenance_time[ID] = (next_fail_time[ID] + pd.to_timedelta(1, unit='D')).replace(hour=6, minute=0)  # maintenance starts only at next_fail +1D but machine can fail before so its for the worst case scenario
        else:
            maintenance_time[ID] = day[ID][0]
    print(next_fail_time[0:3], maintenance_time[0:3])

# calculates fail in fail out
    for mach in range(0, total_mach):
        if maintenance_time[mach] <= in_week_last:
            if maintenance_time[mach] >= current_simul_date1[mach + 1]:
                fail_in[mach] += 1
        elif maintenance_time[mach] >= in_week_last:
            if maintenance_time[mach] <= next_week_last:
                fail_out += 1

# Calculates changes between genes:
    change = 0
    for j in range (0, size):
        for ID in range(0, total_mach):
            if individual[j+ (ID*size)] != individual [j+ (ID*size)-1] and j!=0 :
                change +=1


    up = 0
    for j in range (0, size):
        for ID in range(0, total_mach):
            if individual[j+ (ID*size)] - 0 > 1:
                up +=1
            elif individual[j+ (ID*size)] - 0 < -1:
                up += 1


# Calculates sdev: std of loads in a week in a machine
    ind2 = individual[:]
    size2 = len(ind2)//total_mach
    genes=0
    sdev=0 # std of load in individual(per machine)
    for j in range(0, len(ind2)+1):
        if genes == size2:
            K = ind2[j-genes:j]
            sdev = sdev+ np.std(K)
            genes = 0
        genes += 1


# Calculates production
    for ID in range(0, total_mach):
        j = (ID * size)
        i = 0
        block_after = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
        block_stop = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0])
        while j < (((size - 1) + (ID * size))) + 1:
            prod_fail = 0
            blocks_maint = 0
            blocks_fail = 0
            time_in_fail = next_fail_time[ID] - day[ID][i]  # time passed from beginning  slot j to actual failure
            stop = maintenance_time[ID]- next_fail_time[ID]
            if (time_in_fail) < pd.to_timedelta(2, unit='day'):  # if failure inside slot j or j+1
                if (time_in_fail) >= pd.to_timedelta(0, unit='day'):
                    if time_in_fail <= pd.to_timedelta(1, unit='day'):  # if failure inside time slot j
                        if next_fail_time[ID].strftime("%d") == maintenance_time[ID].strftime(
                                "%d"):  # if maint same day failure
                            z = ((time_in_fail).seconds) / 3600
                            if time_in_fail == pd.to_timedelta(1,unit='D'):
                                z=24
                            if stop < pd.to_timedelta(0):  # if failure after maintenance
                                blocks_fail =  round(z - (((pd.to_timedelta(24,
                                                                                unit='h') - stop).seconds) / 3600))  # blocks of 3h in failure mode
                            elif stop > pd.to_timedelta(0):  # if failure before maintenance
                                blocks_fail = z
                                if blocks_fail == 24:
                                    block_stop[i] = 0
                                else:
                                    block_stop[i] = round((stop.seconds) / 3600)
                            else:  # if failure = maintenance
                                blocks_fail = z
                        else:
                            z = round(((time_in_fail).seconds) / 3600)
                            if time_in_fail == pd.to_timedelta(1, unit='D'):
                                z = 24
                            block_stop[i] = 24-z
                            block_stop[i+1] = round((stop.seconds) / 3600) - (24-z)
                            blocks_fail = z
                    elif time_in_fail == pd.to_timedelta(0, unit='day'):  # if failure beginning of time slot j
                        if next_fail_time[ID].strftime("%d") == maintenance_time[ID].strftime(
                                "%d"):  # if maint same day failure
                            if stop < pd.to_timedelta(0):  # if failure after maintenance
                                blocks_fail = 0  # blocks of 3h in failure mode
                            elif stop > pd.to_timedelta(0):  # if failure before maintenance
                                blocks_fail= 0
                                block_stop[i] = round((stop.seconds) / 3600)
                            else:  # if failure = maintenance
                                blocks_fail = 0
                        else:
                            block_stop[i] = round((stop.seconds) / 3600)
                            blocks_fail = 0
                    else:  # if failure in time slot j+1
                        blocks_fail = (pd.to_timedelta(24, unit='H') - (time_in_fail - pd.to_timedelta(24,unit='H')))  # blocks in failure due to fail next slot
                        blocks_fail = round(((blocks_fail).seconds) / 3600)  # blocks of 3h in failure mode
                prod_fail = (production[ID + 1]* load_prod(individual[j])) * 0.9 * blocks_fail
            m = (day[ID][i + 1] - maintenance_time[ID])  # time passed from start maintenance to slot j+1
            if m > pd.to_timedelta(0, unit='day'):  # if m in slot j
                if m <= pd.to_timedelta(1, unit='day'):  # if m less than mttr round up to days
                    if m >= pd.to_timedelta(mttr[ID + 1] * 3, unit='H'):  # if m bigger or equal to mttr in hours
                        blocks_maint = pd.to_timedelta(mttr[ID + 1] * 3, unit='H')
                        blocks_maint = round(((blocks_maint).seconds) / 3600)
                    else:
                        if m  == pd.to_timedelta(1,unit='D'):
                            blocks_maint = 24
                        else:
                            blocks_maint = round((m.seconds) / 3600)
                        coise = pd.to_timedelta(mttr[ID + 1] * 3, unit='H') - m
                        if coise  == pd.to_timedelta(1,unit='D'):
                            block_after[i + 1] = 24
                        else:
                            block_after[i + 1] = round(((coise).seconds) / 3600)

            prod1 = prod1 + prod_fail + ((production[ID + 1] * load_prod(individual[j]) * (24 - (blocks_fail + blocks_maint + block_after[i]+ block_stop[i]))))
            print('bf, ba,bm, bs', blocks_fail, block_after, blocks_maint, m, block_stop)
            j += 1
            i += 1
    print(prod1)
    fit = (target[0]-prod1)**2 + ((sum(fail_in)) * 1000) + (fail_out * 600) + (sdev * 800) + (change * 500) + (up * 600)
    if target[0]-prod1>0:
        fit=1000000
    print('fit',fit, up, change, sdev)
    return fit,

def load_prod(loadd):
    ind_load = 0
    if loadd == 2: ind_load = 2
    if loadd == 1: ind_load = 1.5
    if loadd == 0: ind_load = 1
    if loadd == -1: ind_load = 0.5
    if loadd == -2: ind_load = 0.25
    return ind_load

def update_maint(maint_df, recent_maint_matrix, ID, current_simul_date, X):
    """Updates the recent_maint_matrix and resets comp age when a maintenance needs to be performed
                Input : maint_df, recent_maint_matrix, ID and current_simul_date
                Returns : X dataframe, recent_maint_matrix and do_maint
        """
    d = maint_df
    d = d[d['machineID'] == ID]
    d = d['comp'].where(d['datetime'] == current_simul_date).dropna()
    d.index = range(len(d))
    if len(d) != 0:
        for i in range(0, len(d)):
            if d[i] == 'comp1':
                X.comp2.loc[(X['datetime'] == current_simul_date) & (X['machineID'] == ID)] = 0
                recent_maint_matrix[0, ID] = 0  # row 0 is for comp1
            elif d[i] == 'comp2':
                X.comp2.loc[(X['datetime'] == current_simul_date ) & (X['machineID'] == ID)] = 0
                recent_maint_matrix[1, ID] = 0 # row 1 is for comp2
            elif d[i] == 'comp3':
                X.comp2.loc[(X['datetime'] == current_simul_date) & (X['machineID'] == ID)] = 0
                recent_maint_matrix[2, ID] = 0
            elif d[i] == 'comp4':
                X.comp2.loc[(X['datetime'] == current_simul_date) & (X['machineID'] == ID)] = 0
                recent_maint_matrix[3, ID] = 0
        do_maint[ID] = 1 # has to do maintenance
    else:
        do_maint[ID] = 0 # no maintenance to perform

    return do_maint, X, recent_maint_matrix

def write_maint(maint_df, maint_date, ID, comp):
    """writes maintenance date to dataframe
                Input : maint_df, maint_date, ID and comp
                Returns : maint_df
        """
    comp_r = comp
    ID_r = ID
    df2 = pd.DataFrame([[maint_date, ID_r, comp_r]], columns=('datetime', 'machineID', 'comp'))
    maint_df = maint_df.append(df2)
    maint_df = maint_df.sort_values(by=['machineID', 'datetime', 'comp'])

    return  maint_df

def program_maint(prediction, ID, maint_df, current_simul_date, recent_maint_matrix):
    """Schedules a maintenance date in case of failure prediction:
                Input : prediction, ID, maint_df, current_simul_date and recent_maint
                Returns : recent_maint matrix and maint_df dataframe
        """
    if prediction[ID - 1] == 'comp2':
        if current_simul_date[ID].time().strftime('%H:%M:%S') < '12:00:00':
            maint_date = ((current_simul_date[ID] + pd.to_timedelta(1, unit='D')).replace(hour=6, minute=0))
            #print(maint_date)
        else:
            maint_date = ((current_simul_date[ID] + pd.to_timedelta(2, unit='D')).replace(hour=6, minute=0))
            #print(maint_date)
        comp ='comp2'
        if recent_maint_matrix[1, ID] == 0:
            maint_df = write_maint(maint_df, maint_date, ID, comp)
            recent_maint_matrix[1, ID] = maint_date
    # elif prediction[ID - 1] == 'comp1':
    #     # calculate maintenance date:
    #     # if before 12am schedule maint for day of failure at 6am
    #     # if after 12 am schedule maint for day after failure at 6am
    #     if current_simul_date[ID].time().strftime('%H:%M:%S') < '12:00:00':
    #         maint_date = ((current_simul_date[ID] + pd.to_timedelta(1, unit='D')).replace(hour=6, minute=0))
    #     else:
    #         maint_date = ((current_simul_date[ID] + pd.to_timedelta(2, unit='D')).replace(hour=6, minute=0))
    #     comp ='comp1'
    #     # write maintenance in matrix [0] - comp1, [1] - comp1...
    #     if recent_maint_matrix[0, ID] == 0:
    #         maint_df = write_maint(maint_df, maint_date, ID, comp)
    #         recent_maint_matrix[0, ID] = maint_date
    # elif prediction[ID - 1] == 'comp3':
    #     if current_simul_date[ID].time().strftime('%H:%M:%S') < '12:00:00':
    #         maint_date = ((current_simul_date[ID] + pd.to_timedelta(1, unit='D')).replace(hour=6, minute=0))
    #     else:
    #         maint_date = ((current_simul_date[ID] + pd.to_timedelta(2, unit='D')).replace(hour=6, minute=0))
    #     comp ='comp3'
    #     if recent_maint_matrix[2, ID] == 0:
    #         maint_df = write_maint(maint_df, maint_date, ID, comp)
    #         recent_maint_matrix[2, ID] = maint_date
    # elif prediction[ID - 1] == 'comp4':
    #     if current_simul_date[ID].time().strftime('%H:%M:%S') < '12:00:00':
    #         maint_date = ((current_simul_date[ID] + pd.to_timedelta(1, unit='D')).replace(hour=6, minute=0))
    #     else:
    #         maint_date = ((current_simul_date[ID] + pd.to_timedelta(2, unit='D')).replace(hour=6, minute=0))
    #     comp ='comp4'
    #
    #     if recent_maint_matrix[3, ID] == 0:
    #         maint_df = write_maint(maint_df, maint_date, ID, comp)
    #         recent_maint_matrix[3, ID] = maint_date

    #print(maint_df)
    return maint_df, recent_maint_matrix

def classifier(X):
    """Runs classifier:
            Input : X dataframe
            Returns : Prediction for each machine: 'none', 'comp1', 'comp2'...
    """
    filename = 'finalized_model.sav'
    pickle_in = open(filename, "rb") #open trained classifier
    my_model = pickle.load(pickle_in)
    testX = pd.get_dummies(X.loc[X['datetime'] == X['datetime'].iloc[-1]].drop(['datetime',
                                                                                'machineID',
                                                                                'failure'], 1))
    prediction = my_model.predict(testX)

    return prediction

def flag3 (total_mach, mean_fail, std_fail, X, current_simul_date, ID, index, prediction):
    """Updates X dataframe when in Failure mode with simulated data
            Input : X dataframe, machine ID, number of machines, index array, current_simul_date, mean and std fail
            Returns : X dataframe, index and current simul date and flag state
    """

    L = labeled_features
    empty_copy = pd.DataFrame(0, columns=L.columns, index=L.index)
    machines['model'] = machines['model'].astype('category')
    x = L['model']
    empty_copy['model'] = x
    empty_copy = empty_copy[0:1]

    fail = pd.DataFrame([[X.machineID[index[0] - total_mach], current_simul_date[ID],
                          np.random.normal(mean_fail[ID - 1, 0], std_fail[ID - 1, 0]),
                          np.random.normal(mean_fail[ID - 1, 1], std_fail[ID - 1, 1])
                             , np.random.normal(mean_fail[ID - 1, 2], std_fail[ID - 1, 2]),
                          np.random.normal(mean_fail[ID - 1, 3], std_fail[ID - 1, 3])
                             , np.random.normal(mean_fail[ID - 1, 4], std_fail[ID - 1, 4]),
                          np.random.normal(mean_fail[ID - 1, 5], std_fail[ID - 1, 5])
                             , np.random.normal(mean_fail[ID - 1, 6], std_fail[ID - 1, 6]),
                          np.random.normal(mean_fail[ID - 1, 7], std_fail[ID - 1, 7])
                             , np.random.normal(mean_fail[ID - 1, 8], std_fail[ID - 1, 8]),
                          np.random.normal(mean_fail[ID - 1, 9], std_fail[ID - 1, 9])
                             , np.random.normal(mean_fail[ID - 1, 10], std_fail[ID - 1, 10]),
                          np.random.normal(mean_fail[ID - 1, 11], std_fail[ID - 1, 11])
                             , np.random.normal(mean_fail[ID - 1, 12], std_fail[ID - 1, 12]),
                          np.random.normal(mean_fail[ID - 1, 13], std_fail[ID - 1, 13])
                             , np.random.normal(mean_fail[ID - 1, 14], std_fail[ID - 1, 14]),
                          np.random.normal(mean_fail[ID - 1, 15], std_fail[ID - 1, 15])
                             , 0, 1, 1, 0, 0, X.comp1[index[0] - total_mach],
                          X.comp2[index[0] - total_mach], X.comp3[index[0] - total_mach]
                             , X.comp4[index[0] - total_mach], X.model[index[0] - total_mach],
                          X.age[index[0] - total_mach], prediction[ID - 1]]], columns=L.columns)
    fail['model'] = empty_copy['model'].astype('category')
    X = X.append(fail, ignore_index=True)
    in_fail[ID] += 1

    # Update Component age
    X.comp1[index[0]] = X.comp1[index[0] - total_mach] + 0.125
    X.comp2[index[0]] = X.comp2[index[0] - total_mach] + 0.125
    X.comp3[index[0]] = X.comp3[index[0] - total_mach] + 0.125
    X.comp4[index[0]] = X.comp4[index[0] - total_mach] + 0.125

    # if failure data injected for 24h return to normal mode of simulated data (flag2):
    if in_fail[ID] >= 16:
        if prediction[ID - 1] == 'comp2':  # when more comps replace with != 'none'
            flag_state[ID] = 0
            in_fail[ID] = 0
        else:
            print('DID NOT PREDICT COMP2 FAILURE')
            sys.exit("DID NOT PREDICT COMP2 FAILURE")

    # update variable each cycle
    current_simul_date[ID] = current_simul_date[ID] + pd.to_timedelta(3, unit='H')
    index[ID] = index[ID] + 1
    index[0] = index[0] + 1

    return X, index, current_simul_date, flag_state

def flag2(total_mach, mean, std, X, current_simul_date, ID, index, in_normal, mtbf, prediction):
    """Updates X dataframe when in Normal mode with simulated data
            Input : X dataframe, machine ID, index array, current_simul_date, in_normal count, mena and std
            Returns : X dataframe, index and current simul date and flag state
    """
    L = labeled_features
    empty_copy= pd.DataFrame(0, columns=L.columns, index=L.index)
    machines['model'] = machines['model'].astype('category')
    x = L['model']
    empty_copy['model'] = x
    empty_copy = empty_copy[0:1]

     # create row for dataframe with sensor from random normal dist fitted to non failure data
    non_fail = pd.DataFrame([[X.machineID[index[0] - total_mach], current_simul_date[ID],
                                  np.random.normal(mean[ID - 1, 0], std[ID - 1, 0]),
                                  np.random.normal(mean[ID - 1, 1], std[ID - 1, 1])
                                     , np.random.normal(mean[ID - 1, 2], std[ID - 1, 2]),
                                  np.random.normal(mean[ID - 1, 3], std[ID - 1, 3])
                                     , np.random.normal(mean[ID - 1, 4], std[ID - 1, 4]),
                                  np.random.normal(mean[ID - 1, 5], std[ID - 1, 5])
                                     , np.random.normal(mean[ID - 1, 6], std[ID - 1, 6]),
                                  np.random.normal(mean[ID - 1, 7], std[ID - 1, 7])
                                     , np.random.normal(mean[ID - 1, 8], std[ID - 1, 8]),
                                  np.random.normal(mean[ID - 1, 9], std[ID - 1, 9])
                                     , np.random.normal(mean[ID - 1, 10], std[ID - 1, 10]),
                                  np.random.normal(mean[ID - 1, 11], std[ID - 1, 11])
                                     , np.random.normal(mean[ID - 1, 12], std[ID - 1, 12]),
                                  np.random.normal(mean[ID - 1, 13], std[ID - 1, 13])
                                     , np.random.normal(mean[ID - 1, 14], std[ID - 1, 14]),
                                  np.random.normal(mean[ID - 1, 15], std[ID - 1, 15])
                                     , 0, 0, 0, 0, 0, X.comp1[index[0] - total_mach], X.comp2[index[0] - total_mach],
                                  X.comp3[index[0] - total_mach]
                                     , X.comp4[index[0] - total_mach], X.model[index[0] - total_mach],
                                  X.age[index[0] - total_mach], prediction[ID-1]]], columns=L.columns)
    non_fail['model'] = empty_copy['model'].astype('category')
    X = X.append(non_fail, ignore_index=True)
    in_normal[ID] += 3

    # Update Component age
    X.comp1[index[0]] = X.comp1[index[0] - total_mach] + 0.125
    X.comp2[index[0]] = X.comp2[index[0] - total_mach] + 0.125
    X.comp3[index[0]] = X.comp3[index[0] - total_mach] + 0.125
    X.comp4[index[0]] = X.comp4[index[0] - total_mach] + 0.125

    # if times in normal mode passes mean time to failure, enter failure data
    if in_normal[ID] >= mtbf[load[ID]+2][ID]:
        flag_state[ID] = 3
        in_normal[ID] = 0


    # update variable each cycle
    current_simul_date[ID] = current_simul_date[ID] + pd.to_timedelta(3, unit='H')
    index[ID] = index[ID] + 1
    index[0] = index[0] + 1

    return X, index, current_simul_date, flag_state

def flag1(ID, X, index, current_simul_date, total_mach, in_maint, return_fmaint):
    """Updates X dataframe when in Maintenance mode
            Input : X dataframe, machine ID, index array, current_simul_date and total_mach
            Returns : X dataframe, index and current simul date
    """
    L = labeled_features
    empty_copy= pd.DataFrame(0, columns=L.columns, index=L.index)
    machines['model'] = machines['model'].astype('category')
    x = L['model']
    empty_copy['model'] = x
    empty_copy = empty_copy[0:1]

    # create row for dataframe with sensor values at zero
    empty_sensor = pd.DataFrame([[X.machineID[index[0] - total_mach], current_simul_date[ID], 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, X.comp1[index[0]-total_mach], X.comp2[index[0]-total_mach], X.comp3[index[0]-total_mach], X.comp4[index[0]-total_mach], X.model[index[0]-total_mach], X.age[index[0] - total_mach], 'none']], columns=L.columns)
    empty_sensor['model']= empty_copy['model'].astype('category')
    X = X.append(empty_sensor, ignore_index=True)

    # Mantain Component age
    X.comp1[index[0]] = X.comp1[index[0] - total_mach]
    X.comp2[index[0]] = X.comp2[index[0] - total_mach]
    X.comp3[index[0]] = X.comp3[index[0] - total_mach]
    X.comp4[index[0]] = X.comp4[index[0] - total_mach]

    # update variables each cycle
    current_simul_date[ID] = current_simul_date[ID] + pd.to_timedelta(3, unit='H')
    index[ID] += 1
    index[0] += 1
    in_maint[ID] += 1 #cycles in maintenance mode
    return_fmaint[ID] = 1
    return X, index, current_simul_date, in_maint, return_fmaint

def flag0(ID, X, index, current_simul_date, simul_dataframe, return_fmaint):
    """Updates X dataframe when with real data
            Input : X dataframe, machine ID, index array, current_simul_date and current_cycle
            Returns : X dataframe, index and current simul date
    """
    if return_fmaint[ID] == 0  :
        X_new = simul_dataframe.loc[simul_dataframe['machineID'] == ID]
        X = X.append(X_new[index[ID]:(index[ID] + 1)], ignore_index=True) # append windows of 3H
    elif return_fmaint[ID] == 1:
        X_new = simul_dataframe.loc[simul_dataframe['machineID'] == ID]
        X = X.append(X_new[index[ID]:(index[ID] + 1)], ignore_index=True)
        X.comp1[index[0]] = X.comp1[index[0] - total_mach] + 0.125
        X.comp2[index[0]] = X.comp2[index[0] - total_mach] + 0.125
        X.comp3[index[0]] = X.comp3[index[0] - total_mach] + 0.125
        X.comp4[index[0]] = X.comp4[index[0] - total_mach] + 0.125

    # update variable each cycle
    current_simul_date[ID] = current_simul_date[ID] + pd.to_timedelta(3, unit='H')
    index[ID] = index[ID] + 1
    index[0] = index[0] + 1

    return X, index, current_simul_date
ll = []
ll2 = []
ll3 = []
# _________________________Main _____________________

while (current_cycle < total_cycles): # total_cycles):

    #udpates flag state acording to load (if load !=0 change to simulated data)
    for ID in range(1, total_mach + 1):
            if flag_state[ID]==0:
                if index[0] > 2:
                    if load[ID]!=0:
                        flag_state[ID] = 2

    for ID in range(1, total_mach+1):
        if flag_state[ID] == 0: #if real data
            X, index, current_simul_date = flag0(ID, X, index, current_simul_date, simul_dataframe, return_fmaint)
        elif flag_state[ID] == 1: # if in maintenance
            X, index, current_simul_date, in_maint, return_fmaint= flag1(ID, X, index, current_simul_date, total_mach, in_maint, return_fmaint)
        elif flag_state[ID] == 2: # if normal mode simulated
            X, index, current_simul_date, flag_state = flag2(total_mach, mean, std, X, current_simul_date, ID, index, in_normal, mtbf, prediction)
        elif flag_state[ID] == 3 : # if failure mode simulated
            X, index, current_simul_date, flag_state= flag3(total_mach, mean_fail, std_fail, X, current_simul_date, ID, index, prediction)

    # calls predictor
    prediction = classifier(X)
    #print(current_simul_date[1])
    #print(flag_state)
    # programs maintenance action
    for ID in range(1, total_mach + 1):
        maint_df, recent_maint_matrix = program_maint(prediction, ID, maint_df, current_simul_date, recent_maint_matrix)
        do_maint, X, recent_maint_matrix = update_maint(maint_df, recent_maint_matrix, ID, (current_simul_date[ID] - pd.to_timedelta(3, unit='H')), X)

    # updates flags
    for ID in range(1, total_mach + 1):
        if do_maint[ID] == 1: # performing maintenance
            flag_state[ID] = 1 # enter maintenance state
            in_maint[ID] = 0    # resets maintenance count
        if flag_state[ID] == 1: # if in maintenance state
            if in_maint[ID] < mttr[ID]: # test if mean time to repair passed
                flag_state[ID] = 1 # if not passed: maintain flag = 1 (maintenance)
            else:   # if passed test load level
                if load[ID] == 0:   # if load=0 return to flag = 0 (real data)
                    flag_state[ID] = 0
                else:               # if load not 0 go to flag = 2 (simulated normal mode)
                    flag_state[ID] = 2
    #print(flag_state)

    # flag that calls GA up
    for ID in range(1, total_mach + 1):
        if prediction[ID - 1] != 'none' and in_ga == 0:
            start_ga = 0
            id_failed = ID

    # calculate production
    if week_comp < 56: # still not passed a week
        for ID in range(1, total_mach + 1):
            if flag_state[ID] == 1:
                prod = prod + 0  # em maint
            elif prediction[ID-1] == 'comp2' and flag_state[ID]!=1: # if not in maintenance    # when more comps replace with != 'none'
                prod = prod + production[ID] * 0.9 * load_prod(load[ID])  # when in failure mode production decreases 5%
            else:
                prod = prod + production[ID] * load_prod(load[ID])
    else: # week complete; reset variables and restart value calc
        week_comp = 0
        prod = 0
        for ID in range(1, total_mach + 1):
            if flag_state[ID] == 1:
                prod = prod + 0  # em maint
            elif prediction[ID - 1] == 'comp2' and flag_state[ID] != 1:
                prod = prod + production[ID]*0.9*load_prod(load[ID]) # when in failure mode production decreases 5%
            else:
                prod = prod + production[ID]*load_prod(load[ID])
        week_count = week_count + 1
    print("Total production:", prod, "in week", week_count)
    week_comp = week_comp + 1
    #starts GA
    if start_ga == 1:
        in_ga = 1
        # define individual size
        w_quo = (56 - week_comp) // 8  # days left in the week
        w_res = ((56 - week_comp) % 8)  # remainder of division
        size = w_quo
        IND_SIZE = size
        poss_loads = [0, 1, 2, -2, -1]

        random.seed(64)

        creator.create("Fitness", base.Fitness,
                       weights=(-1.0,))  # minimizes fit(if (1.0,-1) maximizes 1 objective, min other)
        creator.create("Individual", list, fitness=creator.Fitness)  # type set, fitness attribute

        toolbox = base.Toolbox()
        toolbox.register("attr_item", random.choice, poss_loads)  # attribute generator(random choice of possible loads)
        toolbox.register("individual", tools.initRepeat, creator.Individual,
                         toolbox.attr_item, (IND_SIZE * total_mach))  # initialize individual
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)  # initialize population

        toolbox.register("evaluate", evalKnapsack)  # evaluation function
        toolbox.register("mate", tools.cxTwoPoint)  # 2pointcrossover
        toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.2)  # mutation technique
        toolbox.register("select", tools.selTournament, tournsize=3)  # selection technique

        # FALTA GUARDAR :current_simul_date, id_failed, w_quo, w_res, prod:
        with open('save.pkl', 'wb') as f:
            pickle.dump(w_res, f)
            pickle.dump(w_quo, f)
            pickle.dump(current_simul_date, f)
            pickle.dump(prod, f)
            pickle.dump(id_failed, f)
            pickle.dump(load,f)
        f.close()
        pop, stats, hof = main_GA()
        print('cromossoma otimo',hof)
        elapsed = timeit.default_timer() - start_time  # elapsed time in seconds
        print(elapsed / 60)
        sys.exit("END")
    current_cycle +=1







#X.to_csv('output1.csv', index=False)

