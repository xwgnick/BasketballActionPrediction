import pandas as pd
import numpy as np
import os
import errno
import pandas as pd
import torch
from torch import nn
import numpy as np
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt; plt.rcdefaults()
import matplotlib.pyplot as plt

def check_replace(dataframe, column_name):
    numbers = []
    for element in dataframe[column_name]:
        if element not in numbers:
            numbers.append(element)
    print(numbers)

#def one_hot(dataset, column_num, num_classes):  #(data, int, int)
#        label = {}
#        data_ret = []
#        for vec in dataset:
#            temp = [0]*num_classes
#            temp[int(vec[column_num])] = 1
#            new_vec = []
#            new_vec.extend(vec[0:column_num])
#            new_vec.extend(temp)
#            new_vec.extend(vec[(column_num+1):])
#            data_ret.append(new_vec)
#        return data_ret

def one_hot(dataframe, column_name, num_class):
    target_column = dataframe[column_name].values
    one_hotted_mat = np.zeros((len(target_column), num_class))
    for i in range(len(target_column)):
        one_hotted_mat[i, target_column[i]] = 1
    origin_ind = dataframe.columns.get_loc(column_name)
    dataframe = dataframe.drop(column_name, axis=1)
    for i in range(one_hotted_mat.shape[1]):
        column_new_name = column_name + "_" + str(i)
        column_values = (one_hotted_mat[:, i]).tolist()
        insert_ind = origin_ind + i
        dataframe.insert(loc=insert_ind, column=column_new_name, value=column_values)
    return dataframe

def standarlization(dataframe, column_names):
    features = dataframe[column_names]
    scaler = StandardScaler().fit(features.values)
    features = scaler.transform(features.values)
    dataframe[column_names] = features
    return dataframe, scaler.mean_, scaler.var_

def normalization(dataframe, column_names):
    features = dataframe[column_names]
    features = MinMaxScaler().fit_transform(features.values.astype(float))
    dataframe[column_names] = features
    return dataframe

#def divide_dataset(dataframe, training_rate = 0.8):
#    #divide the training set
#    training_feature_df = dataframe.iloc[: int(dataframe.shape[0] * training_rate),0:-1]
#    training_label_df = dataframe.iloc[:training_feature_df.shape[0],-1]
#    testing_feature_df = dataframe.iloc[int(dataframe.shape[0] * training_rate) + 1:,0:-1]
#    testing_label_df = dataframe.iloc[:testing_feature_df.shape[0],-1]
#    return training_feature_df, training_label_df, testing_feature_df, testing_label_df
def divide_dataset(dataframe, training_rate = 0.8):
    #divide the training set
    training_feature_df = dataframe.iloc[: int(dataframe.shape[0] * training_rate),0:-1]
    training_label_df = dataframe.iloc[:training_feature_df.shape[0],[-1]]
    testing_feature_df = dataframe.iloc[int(dataframe.shape[0] * training_rate) + 1:,0:-1]
    testing_label_df = dataframe.iloc[:testing_feature_df.shape[0],[-1]]
    return training_feature_df, training_label_df, testing_feature_df, testing_label_df

def reverse_dict(dictionary):
    result = {}
    for key in dictionary:
        result[dictionary[key]] = key
    return result
def point_calculate(d):
    if(d.iloc[0]['p1_team_id']==1388):
        duke_status = 'yes'
    else:
        duke_status = 'no'
    duke_pts = []
    opp_pts = []
    pts_diff = []
    duke_tmp = 0
    opp_tmp = 0
    for index, row in d.iterrows():
        if(row['event.descrip']!='made shot' and row['event.descrip']!='made ft'):
            duke_pts.append(duke_tmp)
            opp_pts.append(opp_tmp)
        else:
            if(row['home'] == duke_status):
                if(row['event.descrip'] == 'made ft'):
                    duke_tmp += 1
                    duke_pts.append(duke_tmp)
                    opp_pts.append(opp_tmp)
                    printline = 'Duke: ' +  str(duke_tmp) + '   Opponent: ' + str(opp_tmp)
                else:
                    if(row['ball_y']<= 47):
                        if((row['ball_y']-4.75)*(row['ball_y']-4.75)+(row['ball_x']-25)*(row['ball_x']-25)< 20.25*20.25):
                            duke_tmp += 2
                            duke_pts.append(duke_tmp)
                            opp_pts.append(opp_tmp)
                            printline = 'Duke: ' +  str(duke_tmp) + '   Opponent: ' + str(opp_tmp)
                        else:
                            duke_tmp += 3
                            duke_pts.append(duke_tmp)
                            opp_pts.append(opp_tmp)
                            printline = 'Duke: ' +  str(duke_tmp) + '   Opponent: ' + str(opp_tmp)
                    else:
                        if((row['ball_y']-89.25)*(row['ball_y']-89.25)+(row['ball_x']-25)*(row['ball_x']-25)< 20.25*20.25):
                            duke_tmp += 2
                            duke_pts.append(duke_tmp)
                            opp_pts.append(opp_tmp)
                            printline = 'Duke: ' +  str(duke_tmp) + '   Opponent: ' + str(opp_tmp)
                        else:
                            duke_tmp += 3
                            duke_pts.append(duke_tmp)
                            opp_pts.append(opp_tmp)
                            printline = 'Duke: ' +  str(duke_tmp) + '   Opponent: ' + str(opp_tmp)
            else:
                if(row['event.descrip'] == 'made ft'):
                    opp_tmp += 1
                    duke_pts.append(duke_tmp)
                    opp_pts.append(opp_tmp)
                    printline = 'Duke: ' +  str(duke_tmp) + '   Opponent: ' + str(opp_tmp)
                else:
                    if(row['ball_y']<= 47):
                        if((row['ball_y']-4.75)*(row['ball_y']-4.75)+(row['ball_x']-25)*(row['ball_x']-25)< 20.25*20.25):
                            opp_tmp += 2
                            duke_pts.append(duke_tmp)
                            opp_pts.append(opp_tmp)
                            printline = 'Duke: ' +  str(duke_tmp) + '   Opponent: ' + str(opp_tmp)
                        else:
                            opp_tmp += 3
                            duke_pts.append(duke_tmp)
                            opp_pts.append(opp_tmp)
                            printline = 'Duke: ' +  str(duke_tmp) + '   Opponent: ' + str(opp_tmp)
                    else:
                        if((row['ball_y']-89.25)*(row['ball_y']-89.25)+(row['ball_x']-25)*(row['ball_x']-25)< 20.25*20.25):
                            opp_tmp += 2
                            duke_pts.append(duke_tmp)
                            opp_pts.append(opp_tmp)
                            printline = 'Duke: ' +  str(duke_tmp) + '   Opponent: ' + str(opp_tmp)
                        else:
                            opp_tmp += 3
                            duke_pts.append(duke_tmp)
                            opp_pts.append(opp_tmp)
                            printline = 'Duke: ' +  str(duke_tmp) + '   Opponent: ' + str(opp_tmp)
        pts_diff.append(duke_pts[index] - opp_pts[index])
    d["score_difference"] = pts_diff
    return d

def loaded_prediction(weight_dir, sample, number_label, mode = "one",with_label = True, verbose = True, acc_verbose = True):
    model =  nn.Sequential(nn.Linear(49,256),
                           nn.ReLU(),
                           nn.Dropout(p=0.2),
                           nn.Linear(256,128),
                           nn.ReLU(),
                           nn.Dropout(p=0.2),
                           nn.Linear(128,64),
                           nn.ReLU(),
                           nn.Dropout(p=0.2),
                           nn.Linear(64,7),
                           nn.LogSoftmax(dim=1))
    model.load_state_dict(torch.load(weight_dir))
    model.eval()
    true_label = []
    if mode == "one":
        temp = np.zeros((2,sample.shape[0]))
        temp[0] = sample.values
        temp[1] = sample.values
        test_df = pd.DataFrame(temp)
        if with_label == True:
            test_df_sample = test_df.iloc[:, :-1]
            test_df_label = test_df.iloc[:,-1]
            if verbose == True:
                print("The true label is: ", number_label[test_df_label.iloc[0]])
        else:
            test_df_sample = test_df
    else:
        test_df = pd.DataFrame(sample)
        if with_label == True:
            test_df_sample = test_df.iloc[:, :-1]
            test_df_label = test_df.iloc[:,-1]
            for label in test_df_label.values:
                true_label.append(number_label[label])
            if verbose == True:
                print("The true label: ", true_label)
        else:
            test_df_sample = test_df
    test_sample_tensor = torch.Tensor(test_df_sample.values)
    fs_prediction = model(test_sample_tensor)
    ps = torch.exp(fs_prediction).detach().numpy()
    _, top_class_pred = fs_prediction.topk(1,dim=1)
    if mode == "one":
        prob_temp = []
        i = 0
        for prob in ps[0]:
            prob_temp.append((ps[0][i] + ps[1][i])/2)
            print(number_label)
            print_line = "There are " + str(round(prob_temp[i] * 100,2)) + "% probability that the ball-holder's next move is " + number_label[i]
            if verbose == True:
                print(print_line)
            i += 1
        return prob_temp
        obj = []
        i = 0
        for elements in prob_temp:
            obj.append(number_label[prob_temp.index(elements)])
        plt.bar(range(len(prob_temp)), prob_temp, tick_label = obj)
        plt.show()

        if verbose == True:
            print("Prediction is: ",number_label[prob_temp.index(max(prob_temp))])
            return prob_temp
        else:
            return prob_temp
    else:
        prediction_list = []
        for element in (top_class_pred.numpy()).tolist():
            prediction_list.append(number_label[element[0]])
        if verbose == True:
            print("The prediction: ", prediction_list)
    if with_label == True:
        test_label_tensor = torch.Tensor(test_df_label.values)
        equals = top_class_pred == (test_label_tensor.type(torch.LongTensor)).view(top_class_pred.shape)
        correct_num = torch.sum(equals)
        acc = correct_num.item() / len(equals)
        acc_line = "Accuracy is: " + str(acc * 100) + "%"
        if acc_verbose == True:
            print(acc_line)
    return acc, prediction_list, true_label

def one_hot_decoder(row):
    row_arr = row.values
    return test_arr.index(max(test_arr))

def count_label(label_frame):
    labels = {}
    for element in label_frame:
        if element not in labels:
            labels[element] = 1
        else:
            labels[element] += 1
    return labels
