# BasketballActionPrediction
In this project, I used a 4-layer Recurrent Neural network and a typical 4-layer Neural Network to predict the Basketball players' move. More specifically, the next of of the ball holder on Duke's basketball team. The training data comes from the SportVU dataset of the NCCA season 2014-2015.

## Dataset Cleaning
The row datasets we get is for all the games Duke played in 2014-2015. There are 24 games therefor we rave 24 datasets in our hands. Every dataset is corresponding to one game. The dataset is trying to describe a whole game, so each row of it is a "moment". In one moment, there are more than 60 column including infomation like game clock, shot clock, all the on-court palyers' position(descibed in x, y coordinates), each players' jersey, who is holding the ball and his position on his team, the ball holder's action at that moment etc. For our project, we are trying to predict the ball holders' next action in real-time. We assume that the ball holders' action is related to all the players' position on the court, the game clock, the shot clock, his position in his team, the ball holders' previous actions. So, we kept 49 colums in each row, and deleted other irrelated columns. For some of the rows lacking information in some of its columns like N/A in "game_clock" in some rows, we just simply deleted such rows. 

After we carefully chose the rows and columns in our dataset, we decovered that our dataset is very imbalanced. Players' tend to dribble the ball most of the time in each game. There are more than 40% percent of the datset that the ball holder is "dribbling". So we used the technique of Synthetic Minority Over-sampling Technique (SMOTE) to increase the minority points in the datset. We also did normalization since it is also helpful to deal with the imbalanced datset. Also, to do multi-class classification, we onehotted all the different actions. There are 7 different classes. And we also one hotted some of the categorical data like the jersey of the ball holder. All the non-numerical information are transformed into numerical information too. The datsetset cleaning code is shown below.

```
function test() {
  console.log("notice the blank line before this function?");
}

```
