# InterestRatePredictions-
This was a project  I prepared for the EdX course Analyst's Edge which I worked on this summer.  In this project, I used R in order to predict whether the Federal Reserve would raise interest rates in a given month.  Please see my wiki link for the full write up!

I applied various skills in this assignment: first, consideration of the use of logistic regression versus CART model when solving a classification problem.  It also showed the types of assessments one must make in evaluating superiority of a model - using accuracy and AUC measures.  Finally, I showed consideration for complexity and an understanding of the over-fitting problem, by selecting the appropriate cp value for my CART tree.  


As a former World Banker, the fluctuations in the federal interest rate are always of interest to me.  So I was pleased to bring analytics to this problem.

This was an exam question  I prepared for the EdX course Analyst's Edge which I worked on this summer.  In this project, I used R in order to predict whether the Federal Reserve would raise interest rates in a given month.  
The dataset for this project included monthly economic and political data dating back to the mid-1960's. In this analysis, the dependent variable will be the binary outcome variable RaisedFedFunds, which takes value 1 if the federal funds rate was increased that month and 0 if it was lowered or stayed the same.

Here is a list of the independent variables considered for each month: 
* Date: The date the change was announced.
* Chairman: The name of the Federal Reserve Chairman at the time the change was announced.
* PreviousRate: The federal funds rate in the prior month.
* Streak: The current streak of raising or not raising the rate, e.g. +8 indicates the rate has been increased 8 months in a row, whereas -3 indicates the rate has been lowered or stayed the same for 3 months in a row.
* GDP: The U.S. Gross Domestic Product, in Billions of Chained 2009 US Dollars.
* Unemployment: The unemployment rate in the U.S.
* CPI: The Consumer Price Index, an indicator of inflation, in 
* HomeownershipRate: The rate of homeownership in the u.S.
* DebtAsPctGDP: The U.S. national debt as a percentage of GDP
* DemocraticPres: Whether the sitting U.S. President is a Democrat (DemocraticPres=1) or a Republican (DemocraticPres=0)
* MonthsUntilElection: The number of remaining months until the next U.S. presidential election.

##Approaches: 

This is a classification problem- with two possible classes 1 or 0.  As such, I will test both logistic regression and CART model in order to predict whether interest rates will be raised or not.
Exploring, preparing and splitting the data
After some initial exploration (I looked for missing values, and the distribution of various values).  Now it is time to set the baseline model for accuracy.  I ran the following code: 
table(fedFunds$RaisedFedFunds)

  0   1 
291 294 
> 294/nrow(fedFunds)
[1] 0.5025641
> nrow(fedFunds)
[1] 585

Based on the table above, we see that of the 585 instances of interest rate change, 294 were increases (the other instances are the same or lowered rates).  So our baseline model is to always predict that the Federal Reserve will raise interest rates.  If we do this, we will be correct 50.2% of the time.
For this project, I will split the data into a training and testing set using the caTools package in R.  I will reserve 70% for training and 30% for testing.

##Part 1)  Logistic Regression

Now I will train a logistic regression model using independent variables "PreviousRate", "Streak", "Unemployment", "HomeownershipRate", "DemocraticPres", and "MonthsUntilElection", using the training set to obtain the model.
Based on this initial model, I see that only "Streak" which refers to the number of consecutive months during which the funds have been raised is the ONLY significant variable here (p-value < 0.05).  All others are insignificant.  So, overall, not a highly reliable model.

> logmodel1<-glm(RaisedFedFunds~PreviousRate+Streak+Unemployment+HomeownershipRate+DemocraticPres+MonthsUntilElection, data=train,family="binomial")
> summary(logmodel1)

> Call: glm(formula = RaisedFedFunds ~ PreviousRate + Streak + Unemployment + 
    HomeownershipRate + DemocraticPres + MonthsUntilElection, 
    family = "binomial", data = train)

> Deviance Residuals:     
> Min       1Q   Median       3Q      Max  
> -2.8177  -1.0121   0.2301   1.0491   2.5297  

> Coefficients:
>                      Estimate Std. Error z value Pr(>|z|)    
> (Intercept)          9.121012   5.155774   1.769   0.0769 .  
> PreviousRate        -0.003427   0.032350  -0.106   0.9156    
> Streak               0.157658   0.025147   6.270 3.62e-10 ***
> Unemployment        -0.047449   0.065438  -0.725   0.4684    
> HomeownershipRate   -0.136451   0.076872  -1.775   0.0759 .  
> DemocraticPres       0.347829   0.233200   1.492   0.1358    
> MonthsUntilElection -0.006931   0.007678  -0.903   0.3666    
> Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1
> (Dispersion parameter for binomial family taken to be 1)
> Null deviance: 568.37  on 409  degrees of freedom
> Residual deviance: 492.69  on 403  degrees of freedom
> AIC: 506.69
> Number of Fisher Scoring iterations: 4

Nevertheless, I want to check the accuracy of this model.  So, I use the predict function in order to create predictions on the test set. Then, using a probability threshold of 0.5, I create a confusion matrix for the test set. 
> predicttrain = predict(logmodel1, newdata=train)
> 
> predicttest= predict(logmodel1, newdata=test,type="response")
> 
> table(test$RaisedFedFunds, predicttest> 0.5)
   
    FALSE TRUE
  0    60   27
  1    31   57

Looking diagonally at the true positives and the true negatives (60 + 57) I see that the model will predict the accurate outcome, 67% of the time.  Not bad compared to the baseline of 50%.
Next, I will assess the validity of the model using the Area under the Curve (AUC).  The AUC is the proportion of time the model can differentiate between a randomly selected true positive and true negative. A higher area under the curve is better (means you are capturing more TPs for the same level of FPs -- see the ROC curve below.)  It is better than a simple accuracy calculation done using the confusion matrix because it accounts for probability estimation at any given threshold.  A higher AUC is always better.  
In this case, the AUC is 0.70.
 
ROC Curve: This is a commonly used graph that summarizes the performance of a classifier over all possible thresholds. It is generated by plotting the True Positive Rate (y-axis) against the False Positive Rate (x-axis) as you vary the threshold for assigning observations to a given class. ROC curves show the trade off between false positive and true positive rates

##Part 2) CART Model
Next I will try the CART model to see if it can better predict outcomes than the logistic regression model.
I will use 10-fold cross validation in this example, in order to identify the best cp value for the CART model.  The CP value is like AIC in regression.  It is a complexity indicator.  I measures the trade-off between model complexity and accuracy on the training set.  A smaller cp leads to a bigger tree which runs the risk of overfitting the data.  

I run the following commands in order to obtain the ideal cp value for this CART model.  It is cp=0.16
> set.seed(201)
> 
> numFolds = trainControl(method = "cv", number=10)
> cp.grid = expand.grid( .cp = (0:50)*0.001) 
> train(RaisedFedFunds ~ PreviousRate+Streak+Unemployment+HomeownershipRate+DemocraticPres+MonthsUntilElection, data = train, method = "rpart", trControl = numFolds, tuneGrid = cpGrid )

> RaisedFedFundsTree = rpart(RaisedFedFunds ~ PreviousRate+Streak+Unemployment+HomeownershipRate+DemocraticPres+MonthsUntilElection, data = train, method="class", cp = 0.016)
> 
> 
> 
> prp(RaisedFedFundsTree)
Here is the tree that is generated
 

Then I want to evaluate the accuracy of this CART model.  Is it better than the logistic regression model that I created already?
After creating a confusion matrix for this model, I see that the accuracy is just 0.64%.  

Since the logistic regression is better (67% accuracy) I will stick with that for my final prediction.

##Conclusions: 
This assignment showed various things: first the use of logistic versus CART model for a classification problem.  It also showed the types of assessments one must make in evaluating which model is better - using accuracy and AUC measures.  Finally, I showed consideration for complexity and an understanding of the over fitting problem, by selecting the appropriate cp value for my CART tree.  
