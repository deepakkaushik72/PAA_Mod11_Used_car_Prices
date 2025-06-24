
<ins>Practical Application Assignment II : Used car Price Prediction</ins>

Link to Jupyter Notebook: https://github.com/deepakkaushik72/PAA_Mod11_Used_car_Prices/blob/main/prompt_II.ipynb

Link to the Images: https://github.com/deepakkaushik72/PAA_Mod11_Used_car_Prices/tree/main/images%20-%20Analysis

Link to Data: Cant have the data here as the file size is 54 MB



**<ins>1. BUSINESS PROBLEM:</ins>** 
Predicting the used car prices (Dependent variable is Price) with a clear objective to help the Used Car Dealers to make the Data Driven informed decisions on negotiating the Car Prices and maximize their margins.

I.	The goal is to develop a predictive model that learns the relationship between the input features/variables and the continuous     target variable "used car price" using historical data. Since this is about predicting continuous variable (price), it’s a         "Regression problem". I will be using the Regression techniques to accurately predict the price of the used car with the           following features in the given dataset:

*•	Target Variable: Used Car Price*

*•	Numeric Features: Mileage/Odometer readings, Age of the car (Year)*

*•	Categorical Features: Manufacturer, Model, Cylinders, Condition, Fuel, drive, Transmission Status, type of care etc.*

II.	 This task involves data preprocessing (cleaning, scaling, missing values, duplicates etc..), feature engineering, and              selection, followed by the application of regression techniques (e.g., Linear Regression, Ridge Regression, Lasso Regression)      to identify significant predictors of car prices

III. The model will be trained on a dataset containing historical data of used car sales, including their prices and associated         features.

IV.	The model will be evaluated using metrics like "mean_squared_error" and "R^2" to ensure its accuracy and reliability in            predicting used car prices.

V.	**SUCCESS CRITERIA:** The success of this project will be measured by the model's ability to accurately predict used car             prices, which is:

•	  MSE's of Train and Test datasets.

•	  R^2 values on the Train and validation dataset.

•	  Number and Percentages of car prices predicted within 5%, 10% and 15% Range of Actual prices

**VI.	HYPOTHESIS:** I have the following Hypothesis basis my understanding of the used car market and the features provided in the dataset:

•	  The price of a used car is influenced by its mileage, age, manufacturer, model, condition, fuel type, drive type, and              transmission status.

•	  Cars with lower mileage and newer models are expected to have higher prices.

•	  The condition of the car (e.g., excellent, good, fair, or poor) significantly impacts its price.

•	  Different manufacturers and models have varying price ranges, with some brands being more expensive than others.

**<ins>2.	DATA ANALYSIS and UNDERSTANDING:**</ins>

*•	Used car prices are driven by Age: Prices higher for latest year and goes down as the age goes up, till 25 years. Post 25+         years of age, the prices go up again*

*•	Prices of cars with 50+ years of age till 70 yrs of age goes up again, may be, because of “Rare”, “Classic” status (Not            possible to classify these in Dataset)*

*•	Prices go down as the Odometer reading goes up and this drop is steep initially. However, this is not true for 50+ years    
    category if the car is “Classic” or “Rare”*

*•	The average price is highest for vehicles with title status "Clean"*

*•	"Automatic transmission" vehicles have a higher average price compared to manual transmission vehicles and others*

*•	The average price is highest for vehicles with fuel type "Gas" and "Diesel"*

*•	The average price is highest for vehicles with condition "Good" and "Like New"*

*•	The car price tends to decrease with the increase in odometer reading*

**<ins>3.	DATA PREPARATION**</ins>

*•	Prices of cars data is highly skewed to the right with more than 90% the data in less than 30K price. Mean at around 20K.*

*•	Odometer Data is also highly skewed to the right with mean as 97K *

*•	Removed all the data with Prices as ZERO (33K datapoints from a total of 426K)*

*•	Looked at the Descriptive Statistics of Prices and Odometer and Year(age).*

*•	Removed car prices over $500K (Lot of cars show prices as 111111, 123456 1000000, 9999999 etc… Also, prices of over 10Million      and over 1 Billion as well)*

*•	Base file for consideration has 379K car records*

*•	More than 30% Missing values for columns Cylinders, condition, type, size etc.*

**<ins>4.	FEATURE ENGINEERING</ins>**

*•	Created a Dataframe (vehicles_base_features) for Feature Engineering after including all features with “NAN” values*

*•	Converted Year to Date time and creating a new feature "AGE" from Year*

*•	Created a pipeline and scaled the numeric data and also used OneHotEncoder for categorical features*

*•	Identified 5 Top Features/ independent variables (AGE, Model, Manufacturer, Odometer and Condition) using                          “permutation_importance” from “sklearn.inspection”*

**<ins>5.	MODELING</ins>**

*•	Created a Dataframe (vehicles_base_model) for Modeling and testing various regression models. The dataframe has 223K records       (~60% of original dataframe excluding the ZERO prices data points)*

*•	Created a pipeline with numeric (Age and Odometer) and Categorical (Model, Manufacturer and Condition) features including          scaling of numeric data*

*•	Created a Train, Test Split or Independent Variables (features) and Dependent variable (Price - Log transformed)* 

*•	Identified the best Polynomial degree (degree=5) using the lowest Train and Test MSE (Plotted in the Jupyter notebook)*

*•	Fitted Linear, Ridge and Lasso Regression Models. Hyper Parameter Tuning using GridSearchCV (CV =5)*

**<ins>6.	EVALUATION</ins>**

*•	Ridge Model is the best Model with Hyperparameter Tuning (alpha=0.359). Th train MSE:0.8, Test MSE: 0.10, Train Score: 0.884,      Test Score: 0.853*

*•	Top Ridge Coefficients are plotted in the Jupyter notebook*

*•	The Predicted and Actual prices for both Train and Test data have a good overlapping area (Plotted in the Jupyter notebook)*

**<ins>7.	DEPLOYMENT / RESULTS</ins>**

*•	AGE, MODEL, MANUFACTURER, ODOMETER and CONDITION are the top 5 variables which determines the used car prices*

*•	These features explain 85% of the variance in car Prices (Test R^2 of 0.853)*

*•	Car Prices drops sharply with age initially till 25 yrs (More the age, less the price)* 

*•	Prices tend to go up slightly after 25+ years. For 50+ - 70 years, the prices go up, may be, because of cars being more “Rare”     or “classic” (Need to discuss with the Car Dealer on this spike in prices)* 

*•	Car prices tend to go down with higher Odometer readings (More the Odometer reading, less the price). However, this may not        apply for “Classic” or “Rare” cars*

*•	The average Predicted Prices follow very closely the actual prices for different Odometer ranges (Plotted in Jupyter               notebook)* 

*.  Cars with "New" condition have the highest average price, followed by "Good", "Like New".... "Fair" and "Salvage" conditions       cars have lowest prices.*

*.  Highest variation in Actual and Predicted prices for Cars in "New" condition.*

*•	Model does a great job closely following the actual prices by Manufacturer (Plotted in the Jupyter Notebook)*

*•	**ACCURACY:** 33% of car prices are predicted within 5% range of actual prices, 50% of the car prices are predicted within 10%       range and 63% within 15% range.*

*•	Create a web Interface for Used car Dealers where they can input the Car data and get the Price of the car using this model*

**<ins>8.	NEXT STEPS</ins>**

*•	Car Dealers to validate the Data for Car Prices greater than 500K (Prices such as 111111, 999999, 123456789, and crossing          millions and Billions need revisit/rechecking)* 

*•	CONDITION is one of the top features for predicting car prices and should be included in the car data by Car Dealers               (Currently    missing ~40% data)*

*•	Car Dealers to capture the car status (“Rare”, “Classic” etc) so that it can be used for predicting the prices of these cars.      Currently, this data is not available.*

*•	The Modeling Team need to understand more context on the “Classic”, “Rare” category cars and what drives their prices for cars     with age between 50 to 100 years.*

*•	For further improvement Model Accuracy, the modeling Team can try Deep Learning Models/Neural Networks so improve results          are compared to Regression models*

*•	Jointly work with the Car Dealers on the Deployment Plan*
