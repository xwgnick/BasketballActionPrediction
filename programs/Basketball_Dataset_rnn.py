import os
import numpy as np
import pandas as pd
from Basketball_set_funcitons import check_replace, one_hot, standarlization, normalization, divide_dataset, reverse_dict, point_calculate

class Dataset:
    dataset_temp_dir = None
    name = None
    origin = None
    labelled = None
    OneHotted = None
    transformed = None
    nn_ready = None
    label_number = {'touch': 0, 'dribble': 1, 'pass': 2, 'missed shot': 3, 'offensive rebound': 4, 'made shot': 3, 'blocked shot': 3, 'defensive rebound': 4, 'turnover': 5, 'assist': 2}
    real_label_number = {'touch': 0, 'dribble': 1, 'pass': 2, "shot": 3, "rebound":4, "turnover":5}
    mean_list = None
    var_list = None

    def replace_string(self, dictionary, column_name):
        self.labelled[column_name].replace(to_replace=dictionary, inplace=True)

    def initialize_dir(self):
        cwd = os.getcwd()
        self.dataset_temp_dir = cwd + "/../" + "dataset_temp/"
        if not os.path.exists(self.dataset_temp_dir):
            try:
                os.makedirs(self.dataset_temp_dir)
            except OSError as e:
                if e.errno != errno.EEXIST:
                    raise

    def __init__(self, directory, dataset_name, home_visit, shuffle = False, choose_duke = True, standardize = False, normalize = True, save = False):
        self.initialize_dir()
        dataset_org = pd.read_csv(directory)
        self.name = dataset_name

        #add point difference
        self.origin = dataset_org.copy()
        self.origin = point_calculate(dataset_org)
        #drop the columns and rows we don't need
        self.labelled = self.origin.drop(columns=["name","global.player.id","game_id", "time", "event.id","p1_team_id", "p10_team_id", "p1_global_id", "p2_global_id", "p3_global_id", "p4_global_id", "p5_global_id", "p6_global_id", "p7_global_id", "p8_global_id", "p9_global_id", "p10_global_id"])
        self.labelled = self.labelled.drop(self.labelled.columns[0], axis=1)
        self.labelled = self.labelled[self.labelled["event.descrip"] != 'EOP']
        self.labelled = self.labelled[self.labelled["event.descrip"] != 'nan']
        self.labelled = self.labelled[self.labelled["event.descrip"] != 'none']
        self.labelled = self.labelled[self.labelled["event.descrip"] != 'missed ft']
        self.labelled = self.labelled[self.labelled["event.descrip"] != 'made ft']
        self.labelled = self.labelled[self.labelled["p.poss"] != 0]
        self.labelled = self.labelled.dropna()
        save_line_droped = self.dataset_temp_dir + dataset_name + "_after_dropping.csv"
        self.labelled.to_csv(save_line_droped)
        #set G = 0, F = 1, G/F = 2, C = 3, F/C = 4
        position_number = {"G":0, "F":1, "G/F":2, "C":3, "F/C":4}
        self.replace_string(position_number, "position")

        #For "home", set no = 0, yes = 1
        home_number = {"no":0, "yes":1}
        self.replace_string(home_number, "home")

        #For "label", build its dictionary, set them to be numbers
        label = []
        for element in self.labelled["event.descrip"]:
            label.append(element)
        self.replace_string(self.label_number, "event.descrip")

        #subtract one for all the elements in "p.poss"
        self.labelled["p.poss"] = self.labelled["p.poss"] - 1

        #add label to the last row
        label = label[1:]
        self.labelled = self.labelled.iloc[:-1]
        self.labelled["label"] = label
        self.replace_string(self.label_number, "label")
        save_line_labelled = self.dataset_temp_dir + dataset_name + "_after_labeling.csv"
        if save == True:
            self.labelled.to_csv(save_line_labelled)
        self.OneHotted = self.labelled
        self.OneHotted = one_hot(self.OneHotted, "event.descrip", 7)
        self.OneHotted = one_hot(self.OneHotted, "position", 5)
        self.OneHotted = one_hot(self.OneHotted, "p.poss", 10)

        #delete the first index column
        save_line_one_hot = self.dataset_temp_dir + dataset_name + "_after_one_hot.csv"
        if save == True:
            self.OneHotted.to_csv(save_line_one_hot)

        #standardize, normalize the dataset
        columns_need_transformation = ["game.clock","shot.clock","p1_x","p2_x","p3_x","p4_x","p5_x","p6_x","p7_x","p8_x","p9_x","p10_x","p1_y","p2_y","p3_y","p4_y","p5_y","p6_y","p7_y","p8_y","p9_y","p10_y","ball_x","ball_y","ball_z"]
        save_line = None
        if standardize == True:
            self.transformed, mean_list, var_list = standarlization(self.OneHotted, columns_need_transformation)
            save_line = self.dataset_temp_dir + dataset_name + "_after_standardized.csv"
        if normalize == True:
            self.transformed = normalization(self.OneHotted, columns_need_transformation)
            save_line = self.dataset_temp_dir + dataset_name + "_after_normalized.csv"
        if save == True:
            self.transformed.to_csv(save_line)

        #shuffle dataframe
        if shuffle == True:
            self.nn_ready = self.transformed.sample(frac=1).reset_index(drop=True)
        else:
            self.nn_ready = self.transformed

        #choose the duke team
        if choose_duke == True:
            self.nn_ready = self.nn_ready[self.nn_ready["home"] == home_visit]
        printline = "dataset" + dataset_name + " is imported."
        save_line_ready =self.dataset_temp_dir + dataset_name + "ready.csv"
        if save == True:
            self.nn_ready.to_csv(save_line_ready)
        print(printline)
