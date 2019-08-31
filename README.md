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

## Model Choosing
We first chose a typical 4-layer neural network to train the model without balancing the data. It turns out that the result is very imbalanced. Although the testing accuray reached 68%, it is easy to see that our model is trying to predict every thing to be "dribble". By observing that our dataset is acatually a sequence since all the rows are ordered by time and the information in each row is affected by all the rows before it. So we started to consider maybe RNN will do a better job. To set a contrast group, we also kept our origin 4-layer NN model. So, we chose a typical 4-layer NN and a 4-layer RNN as our final model.

## Model Training
We built and trained our model in pytorch. Thanks to youtuber @sentdex, I learned how to build and train a RNN in Pytorch. The model building code of the 4-layer NN is shown below:

```
        self.temp_model = nn.Sequential(nn.Linear(49,256),
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
# 

```
The model building code of  the 4-layer RNN is shown below:


```

    model = Sequential()
    model.add(LSTM(128, input_shape=((SEQ_LEN, 49)), return_sequences=True))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())  #normalizes activation outputs, same reason you want to normalize your input data.

    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.1))
    model.add(BatchNormalization())

    model.add(LSTM(128))
    model.add(Dropout(0.2))
    model.add(BatchNormalization())

    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(6, activation='softmax'))


```

I also used Cross-validation technique to choose the models with highest validation accuracy during training process. The code is shown below:

```

    def cross_fit(self, epoch = 200, smote = False):
        self.chosen_models = []
        for i in range(len(self.datasets_training) - 1):
#        for i in range(1):
            print("********************************************************************")
            printline = "cross vadation processing the " + str(i) + " time"
            print(printline)
            self.build_model()
            exculde_list = [i, i + 1]
            validation_sets = [self.datasets_training[i], self.datasets_training[i + 1]]
            training_sets = [elem for j, elem in enumerate(self.datasets_training) if j not in exculde_list]
            training_feature_combine, training_label_combine, _,_ = self.dataset_combining(training_sets)
            if smote == True:
                label_num_org = count_label(training_label_combine["label"])
                print("The label number before smote is: ", label_num_org)
                sm = SMOTE(sampling_strategy = {2:label_num_org[1]})
                X_res, y_res = sm.fit_resample(training_feature_combine, training_label_combine["label"])
                training_feature_combine = pd.DataFrame(X_res)
                training_label_combine = pd.DataFrame(y_res)
                training_label_combine=training_label_combine.rename(columns = {0:"label"})
                label_num_smoted = count_label(training_label_combine["label"])
                print("The label number after somte is: ", label_num_smoted)
            testing_feature_combine, testing_label_combine, _,_ = self.dataset_combining(validation_sets)
            self.chosen_models.append(self.train(training_feature_combine, training_label_combine, testing_feature_combine, testing_label_combine, epoch = epoch))
        highest_acc = 0
        chosen_model = None
        testing_sets = []
        for dataset in self.datasets_testing:
            testing_sets.append(dataset)
        if len(testing_sets) > 1:
            testing_set = pd.concat(testing_sets)
        else:
            testing_set = testing_sets[0]
        for model in self.chosen_models:
            sample = testing_set.nn_ready.iloc[:]
            acc, _, _ = self.prediction(model, sample, reverse_dict(self.datasets_training[0].real_label_number), mode = "more", verbose = False, acc_verbose = False)
            if acc > highest_acc:
                highest_acc = acc
                self.model = model


```

## Testing Result
1. Testing Accuracy:
Testing accuracy of the typical sequential neural network: 75.85%
Testing accuracy of the Recurrent neural network: 76.51%
2. Error Analysis -- See which class is the most difficult to predict
<img width="321" alt="Screen Shot 2019-08-31 at 3 42 21 PM" src="https://user-images.githubusercontent.com/47826970/64068426-fa6f1f00-cc05-11e9-80c0-60bfdb62eefc.png">
3. Confusion Matrix -- Judge the prediction performance
<img width="675" alt="Screen Shot 2019-08-31 at 3 42 27 PM" src="https://user-images.githubusercontent.com/47826970/64068433-02c75a00-cc06-11e9-97c1-d05a9fc0e2f2.png">

## Conclusion

1. Imbalanced data can make the prediction result very biased. In our model,
more than 40% of the actions made by the ball holder are dribble. So, our
model learned has the trend to predict other classes to be dribble more
likely. Using normalization and SMOTE can decrease the influence brought
by the imbalanced data.
2. RNN has better performance on sequential data. It improved the testing
accuracy of our model from 75.85% to 76.51%. More importantly, RNN
improved our model’s performance on imbalanced dataset. It decreased the
error rate of “pass” (the most imbalanced class in our dataset) from 77% to
71%.
3. The reason RNN did a better performance on the minority classes (pass) is
because it sacrificed some of the prediction accuracy on the majority class
(dribble). In the confusion matrix, the correct labelled “dribble” decreased
from 569 to 562 while the correct labelled “pass” increased from 49 to 71.
