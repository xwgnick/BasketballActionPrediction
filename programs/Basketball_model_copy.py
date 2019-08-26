import pandas as pd
from Basketball_Dataset_rnn import Dataset
import torch
from torchvision import datasets, transforms
from torch import nn, optim
from imblearn.over_sampling import SMOTE
from Basketball_set_funcitons import reverse_dict, divide_dataset, count_label
import numpy as np

class Basketball_model:
    temp_model = None
    chosen_models = []
    model = None
    datasets_testing = None
    datasets_training = None

    def dataset_importing(self, dataset_dir):
        self.datasets_testing = []
        self.datasets_training = []
        self.datasets_training.append(Dataset(dataset_dir + 'dataset_20141114.csv', "20141114", home_visit = 1, standardize = False, normalize = True))
        self.datasets_training.append(Dataset(dataset_dir + 'dataset_20141118.csv', "20141118", home_visit = 1, standardize = False, normalize = True))
        self.datasets_training.append(Dataset(dataset_dir + 'dataset_20141121.csv', "20141121", home_visit = 1, standardize = False, normalize = True))
        self.datasets_training.append(Dataset(dataset_dir + 'dataset_20141122.csv', "20141122", home_visit = 1, standardize = False, normalize = True))
        self.datasets_training.append(Dataset(dataset_dir + 'dataset_20141126.csv', "20141126", home_visit = 1, standardize = False, normalize = True))
        self.datasets_training.append(Dataset(dataset_dir + 'dataset_20141130.csv', "20141130", home_visit = 1, standardize = False, normalize = True))
        self.datasets_training.append(Dataset(dataset_dir + 'dataset_20141215.csv', "20141215", home_visit = 1, standardize = False, normalize = True))
        self.datasets_training.append(Dataset(dataset_dir + 'dataset_20141229.csv', "20141229", home_visit = 1, standardize = False, normalize = True))
        self.datasets_training.append(Dataset(dataset_dir + 'dataset_20141231.csv', "20141231", home_visit = 1, standardize = False, normalize = True))
        self.datasets_training.append(Dataset(dataset_dir + 'dataset_20150103.csv', "20150103", home_visit = 1, standardize = False, normalize = True))
        self.datasets_training.append(Dataset(dataset_dir + 'dataset_20150113.csv', "20150113", home_visit = 1, standardize = False, normalize = True))
        self.datasets_training.append(Dataset(dataset_dir + 'dataset_20150117.csv', "20150117", home_visit = 0, standardize = False, normalize = True))
        self.datasets_training.append(Dataset(dataset_dir + 'dataset_20150119.csv', "20150119", home_visit = 1, standardize = False, normalize = True))
        self.datasets_training.append(Dataset(dataset_dir + 'dataset_20150125.csv', "20150125", home_visit = 0, standardize = False, normalize = True))
        self.datasets_training.append(Dataset(dataset_dir + 'dataset_20150204.csv', "20150204", home_visit = 1, standardize = False, normalize = True))
        self.datasets_training.append(Dataset(dataset_dir + 'dataset_20150207.csv', "20150207", home_visit = 1, standardize = False, normalize = True))
        self.datasets_training.append(Dataset(dataset_dir + 'dataset_20150218.csv', "20150218", home_visit = 1, standardize = False, normalize = True))
        self.datasets_training.append(Dataset(dataset_dir + 'dataset_20150221.csv', "20150221", home_visit = 1, standardize = False, normalize = True))
        self.datasets_training.append(Dataset(dataset_dir + 'dataset_20150228.csv', "20150228", home_visit = 1, standardize = False, normalize = True))
        self.datasets_training.append(Dataset(dataset_dir + 'dataset_20150304.csv', "20150304", home_visit = 1, standardize = False, normalize = True))
        self.datasets_testing.append(Dataset(dataset_dir + '/dataset_20150320.csv', "20150320", home_visit = 1, shuffle = False, standardize = False, normalize = True, save = True))

    def build_model(self):
#        self.temp_model = nn.Sequential(nn.Linear(49,128),
#                              nn.ReLU(),
#                              nn.Dropout(p=0.2),
#                              nn.Linear(128,64),
#                              nn.ReLU(),
#                              nn.Dropout(p=0.2),
#                              nn.Linear(64,7),
#                              nn.LogSoftmax(dim=1))
        self.temp_model = nn.Sequential(nn.Linear(49,256),
                              nn.ReLU(),
                              nn.Dropout(p=0.2),
                              nn.Linear(256,128),
                              nn.ReLU(),
                              nn.Dropout(p=0.2),
                              nn.Linear(128,64),
                              nn.ReLU(),
                              nn.Dropout(p=0.2),
                              nn.Linear(64,6),
                              nn.LogSoftmax(dim=1))
#        self.temp_model = nn.Sequential(nn.Linear(49,512),
#                                        nn.ReLU(),
#                                        nn.Dropout(p=0.4),
#                                        nn.Linear(512,256),
#                                        nn.ReLU(),
#                                        nn.Dropout(p=0.4),
#                                        nn.Linear(256,128),
#                                        nn.ReLU(),
#                                        nn.Dropout(p=0.4),
#                                        nn.Linear(128,64),
#                                        nn.ReLU(),
#                                        nn.Dropout(p=0.4),
#                                        nn.Linear(64,7),
#                                        nn.LogSoftmax(dim=1))
    def __init__(self, dataset_dir):
        self.build_model()
        self.dataset_importing(dataset_dir)

    def dataset_combining(self, datasets):
        training_feature_combine_list = []
        training_label_combine_list = []
        testing_feature_combine_list = []
        testing_label_combine_list = []
        for dataset in datasets:
            training_feature_df, training_label_df, testing_feature_df, testing_label_df = divide_dataset(dataset.nn_ready, 0.95)
            training_feature_combine_list.append(training_feature_df)
            training_label_combine_list.append(training_label_df)
            testing_feature_combine_list.append(testing_feature_df)
            testing_label_combine_list.append(testing_label_df)
        training_feature_combine = pd.concat(training_feature_combine_list)
        training_label_combine = pd.concat(training_label_combine_list)
        testing_feature_combine = pd.concat(testing_feature_combine_list)
        testing_label_combine = pd.concat(testing_label_combine_list)
        return training_feature_combine, training_label_combine, testing_feature_combine, testing_label_combine

    def train(self, training_feature_df, training_label_df, testing_feature_df, testing_label_df, epoch = 100, criterion = nn.NLLLoss(), learning_rate = 0.0001):
        #set up the trainloader
        train = torch.utils.data.TensorDataset(torch.Tensor(training_feature_df.values), (torch.Tensor(training_label_df["label"].values)).type(torch.LongTensor))
        # train = torch.utils.data.TensorDataset(torch.Tensor(training_feature_df.values), (torch.Tensor(training_label_df.values)).type(torch.LongTensor))
        train_loader = torch.utils.data.DataLoader(train, batch_size = 128, shuffle = True)
        test = torch.utils.data.TensorDataset(torch.Tensor(testing_feature_df.values), (torch.Tensor(testing_label_df["label"].values)).type(torch.LongTensor))
        test_loader = torch.utils.data.DataLoader(test, batch_size = 128, shuffle = True)

        optimizer = optim.Adam(self.temp_model.parameters(), lr=learning_rate)
        epochs = epoch
        train_losses, test_losses = [],[]
        num_train_samples = training_feature_df.shape[0]
        num_test_smaples = testing_feature_df.shape[0]
        higest_val_acc = 0
        chosen_model = None
        for i in range(epochs):
            print("---------------------------------------")
            train_batch_correct_num = 0
            test_batch_correct_num = 0
            running_loss = 0
            for images, labels in train_loader:
        #         images = images.view(images.shape[0],-1)
                optimizer.zero_grad()
                output = self.temp_model(images)
                ps_train = torch.exp(output)
                top_value_train, top_class_train = ps_train.topk(1,dim=1)
                equals_train = top_class_train == labels.view(top_class_train.shape)
                train_batch_correct_num += torch.sum(equals_train)
                loss = criterion(output, labels)
                running_loss += loss.item()
                loss.backward()
                optimizer.step()

        #     else:
            test_loss = 0
            accuracy = 0
            with torch.no_grad():
                self.temp_model.eval()
                for image, label in test_loader:
                    log_ps = self.temp_model(image)
                    test_loss_batch = criterion(log_ps, label)
                    test_loss += test_loss_batch.item()
                    ps = torch.exp(log_ps)
                    top_value, top_class = ps.topk(1,dim=1)
                    equals = top_class == label.view(top_class.shape)
                    test_batch_correct_num += torch.sum(equals)
        #                 accuracy += torch.mean(equals.type(torch.FloatTensor))

            self.temp_model.train()

            train_losses.append(running_loss/len(train_loader))
            test_losses.append(test_loss/len(test_loader))
            testing_acc = test_batch_correct_num.item() / num_test_smaples
            training_acc = train_batch_correct_num.item() / num_train_samples
            print("Epoch: {}/{}.. ".format(i+1, epochs),
                  "Training Accuracy: {:.3f}".format(training_acc),
                  "Training Loss: {:.3f}.. ".format(train_losses[-1]),
                  "Validation Loss: {:.3f}.. ".format(test_losses[-1]),
                  "Validation Accuracy: {:.3f}".format(testing_acc))
            if testing_acc > higest_val_acc:
                chosen_model = self.temp_model
                higest_val_acc = testing_acc
        return chosen_model
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
    def save(self, dir):
        torch.save(self.model.state_dict(), dir)

    def prediction(self, model, sample, number_label, mode = "one",with_label = True, verbose = True, acc_verbose = True):
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
                test_df_sample = two_test_df
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
                 test_df_sample = two_test_df
        test_sample_tensor = torch.Tensor(test_df_sample.values)
        fs_prediction = model(test_sample_tensor)
        ps = torch.exp(fs_prediction).detach().numpy()
        _, top_class_pred = fs_prediction.topk(1,dim=1)
        if mode == "one":
            prob_temp = []
            i = 0
            for prob in ps[0]:
                prob_temp.append((ps[0][i] + ps[1][i])/2)
                print_line = "There are " + str(round((ps[0][i] + ps[1][i])/2 * 100,2)) + "% probability that the ball-holder's next move is " + number_label[i]
                if verbose == True:
                    print(print_line)
                i += 1
            if verbose == True:
                print("Prediction is: ",number_label[prob_temp.index(max(prob_temp))])
            return
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

    def model_evaluate(self, sample):
        _, prediction_list, true_label = self.prediction(self.model, sample, reverse_dict(self.datasets_training[0].real_label_number), mode = "more", verbose = False, acc_verbose = False)
        diff_label_number = {'touch': 0, 'dribble': 0, 'pass': 0, "shot": 0, "rebound":0, "turnover":0}
        erro_rate = {'touch': 0, 'dribble': 0, 'pass': 0, "shot": 0, "rebound":0, "turnover":0}
        for i in range(len(prediction_list)):
            if prediction_list[i] != true_label[i]:
                diff_label_number[true_label[i]] += 1
        erro_num = sum(diff_label_number.values())
        for label in diff_label_number:
            erro_rate[label] = diff_label_number[label] / erro_num
        for element in erro_rate:
            printline = "The error rate of " + element + " is " + str(erro_rate[element])
            print(printline)
        return erro_rate
