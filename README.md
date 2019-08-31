# BasketballActionPrediction
In this project, I used a 4-layer Recurrent Neural network and a typical 4-layer Neural Network to predict the Basketball players' move. More specifically, the next of of the ball holder on Duke's basketball team. The training data comes from the SportVU dataset of the NCCA season 2014-2015.

## Dataset Cleaning
The row datasets we get is for all the games Duke played in 2014-2015. There are 24 games therefor we rave 24 datasets in our hands. Every dataset is corresponding to one game. The dataset is trying to describe a whole game, so each row of it is a "moment". In one moment, there are more than 60 column including infomation like game clock, shot clock, all the on-court palyers' position(descibed in x, y coordinates), each players' jersey, who is holding the ball and his position on his team, the ball holder's action at that moment etc. For our project, we are trying to predict the ball holders' next action in real-time. We assume that the ball holders' action is related to all the players' position on the court, the game clock, the shot clock, his position in his team, the ball holders' previous actions. So, we kept 49 colums in each row, and deleted other irrelated columns. For some of the rows lacking information in some of its columns like N/A in "game_clock" in some rows, we just simply deleted such rows. 

After we carefully chose the rows and columns in our dataset, we decovered that our dataset is very imbalanced. Players' tend to dribble the ball most of the time in each game. There are more than 40% percent of the datset that the ball holder is "dribbling". So we used the technique of Synthetic Minority Over-sampling Technique (SMOTE) to increase the minority points in the datset. We also did normalization since it is also helpful to deal with the imbalanced datset. Also, to do multi-class classification, we onehotted all the different actions. There are 7 different classes. And we also one hotted some of the categorical data like the jersey of the ball holder. All the non-numerical information are transformed into numerical information too. The datsetset cleaning code is shown below.

```
        self.origin = dataset_org.copy()
        self.origin = point_calculate(dataset_org)
        #drop the columns and rows we don't need
        self.labelled = self.origin.drop(columns=["name","global.player.id","game_id", "time", "event.id","p1_team_id", "p10_team_id", "p1_global_id", "p2_global_id", "p3_global_id", "p4_global_id", "p5_global_id", "p6_global_id", "p7_global_id", "p8_global_id", "p9_global_id", "p10_global_id"])
        self.labelled = self.labelled.drop(self.labelled.columns[0], axis=1)
        self.labelled = self.labelled[self.labelled["event.descrip"] != 'EOP']
        self.labelled = self.labelled[self.labelled["event.descrip"] != 'nan']
        self.labelled = self.labelled[self.labelled["event.descrip"] != 'none']
        self.labelled = self.labelled[self.labelled["p.poss"] != 0]
        self.labelled = self.labelled.dropna()

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
        self.labelled.to_csv(save_line_labelled)
        self.OneHotted = self.labelled
        self.OneHotted = one_hot(self.OneHotted, "event.descrip", 7)
        self.OneHotted = one_hot(self.OneHotted, "position", 5)
        self.OneHotted = one_hot(self.OneHotted, "p.poss", 10)

        #delete the first index column
        save_line_one_hot = self.dataset_temp_dir + dataset_name + "_after_one_hot.csv"
        self.OneHotted.to_csv(save_line_one_hot)

        #standardize the dataset
        columns_need_standarlization = ["game.clock","shot.clock","p1_x","p2_x","p3_x","p4_x","p5_x","p6_x","p7_x","p8_x","p9_x","p10_x","p1_y","p2_y","p3_y","p4_y","p5_y","p6_y","p7_y","p8_y","p9_y","p10_y","ball_x","ball_y","ball_z"]
        self.standardized, mean_list, var_list = standarlization(self.OneHotted, columns_need_standarlization)
        save_line_standardized =self.dataset_temp_dir + dataset_name + "_after_standardized.csv"
        self.standardized.to_csv(save_line_standardized)

        #shuffle dataframe
        if shuffle == True:
            self.nn_ready = self.standardized.sample(frac=1).reset_index(drop=True)
        else:
            self.nn_ready = self.standardized

        #choose the duke team
        if choose_duke == True:
            self.nn_ready = self.nn_ready[self.nn_ready["home"] == home_visit]
        printline = "dataset" + dataset_name + " is imported."
        save_line_ready =self.dataset_temp_dir + dataset_name + "ready.csv"
        self.nn_ready.to_csv(save_line_ready)
        print(printline)

```
