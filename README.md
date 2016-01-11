#DetectaRobo Contest - Winning Solution

This repo contains source code for the winning solution of the DetectaRobo contest. The DetectaRobo contest was held on June 06, 2015 by the Federal Trade Commission (FTC). The challenge was to analyze call log data and predict if the call was a robocall. The FTC provied training and test data: the training data consisted of 250,000 calls with four fields: (1) Call From Number, (2) Call To Number, (3) Date-Time of Call and (4) Robocall (1 for robocalls, 0 for non-robocalls). The test data consisted of 250,000 calls with only the first three fields provided.  My team, team HaV, won first place in this competition. For more information, please visit the [contest website](https://www.ftc.gov/news-events/contests/detectarobo).  

Here is the writeup of our method which we submitted to the FTC:  

>We performed feature engineering on the raw features to extract three types of new features: 
>1) Temporal Information: Day of month, day of week, hour-, minute- and second-of-day, quarter of the hour, time of day (morning-afternoon-evening-night)
>2) Characteristics of originating/receiving phone numbers: Indicator variables showing if the originating number has the same area code, exchange code or state as the receiving number, originating and receiving number state  
>3) Calling Patterns: Volume of calls made from originating number, average time gap between subsequent calls for originating number  
>  
>We believe that complex interactions between these features can predict robocalls. Gradient boosting is a powerful machine learning algorithm that can capture these interactions without overfitting the training data. We use the Python implementation of XGBoost. Using the features described above, we trained a booster with parameters:  
>  
>max_depth=9 (grows trees to this depth)
>eta=0.01 (small value makes booster learn slowly) 
>min_child_weight=15 (roughly corresponds to minimum number of samples need to be in each leaf of the trees. Prevents overly large trees)
>gamma=0.1 (minimum loss reduction to make a further partition on a leaf node of the tree. By keeping larger than 0, prevents overly large trees) 
>  
>Other than max_depth, the values of the remaining parameters are chosen to reduce overfitting. Other parameters are kept at their default values. The booster was run for 3000 iterations.


The training and test data are not public, and hence are not provided.  

Description of files:  
*cleanraw_test.py/cleanraw_train.py*: Takes the raw data, performs feature engineering and outputs the cleaned training/test data to respective csv files.  

*train.py*: Trains the booster on the cleaned training data. Saves the "model" as a Python pickle object to disk.  

*test.py*: Uses the model trained from *train.py* on the cleaned test data to generate class probabiltiies, and saves them to a csv file.  

*final.py*: Uses the class probabilities from *test.py* to make class predictions. 




