<h1>
      AI Precipitation Prediction Project
</h1>
 
<h2>Overview</h2>
<body>This repository contains code for an AI-powered precipitation prediction project. The project includes two main models, Hourly Model and Daily Model, designed to predict precipitation based on different datasets and methodologies.</body>

<h2> Table of Contents</h2>
<body><ul>
      <li>Introduction</li>
       <li>Models</li>
       <li>Hourly Model</li>
       <li>Daily Model</li>
       <li>Data Splitting and Model Validation</li>
       <li>API Integration</li>
       <li>Usage</li>
       <li>Contributing</li>
       <li>License</li>
</ul>
</body>
<h2>Introduction</h2>
<body>Our project focuses on building a sophisticated weather prediction and analysis system that integrates machine learning algorithms, real-time weather data retrieval, and data analysis techniques to forecast and analyze weather conditions accurately. Rainfall is a critical component of our ecosystem. Rainfall influences agriculture, water resource management, urban planning, and disaster preparedness. The ability to accurately predict rainfall helps with daily planning as well as long-term sustainability. For this project specifically, we wanted to create a model to predict rainfall but also make a recommendation on whether to bring an umbrella or not based on the rainfall prediction.

Rainfall patterns can vary significantly making it challenging to accurately predict. Changes in weather and precipitation are factors that contribute to the difficulties in predicting rainfall. Our goal with this project is to provide an accurate way to predict rainfall. This model will use historical weather data collected over the past 20 years and uses methods to preprocess this data. Machine learning algorithms are employed to train the model. The model then learns patterns and relationships between various weather parameters and rainfall levels. Current weather data is then inputted into the model which allows it to make predictions about future rainfall.​</body>
<h2>Models</h2>
<h3>Daily Model</h3>
The first dataset we found was from the National Center for Environmental Information. For our model, we looked at data from the past 20 years. This is the dataset we originally worked with but after some experimentation, we found that it was limited in the weather data it offered.​


<a href= "https://www.ncdc.noaa.gov/cdo-web/search">This is the link to the daily dataset.</a>

<h3>Hourly Model</h3>
For the hourly mode, we used a data set that is from Iowa State University. This dataset offered data every hour and we collected data from the past 10 years​. This dataset allowed us to select more weather data to make our model more robust.

<a href= '[https://mesonet.agron.iastate.edu/request/download.phtml​'](https://mesonet.agron.iastate.edu/request/download.phtml)>This is the link for the hourly dataset. </a>

<h2>Preprocessing</h2>
<h3>Daily Model</h3>
      Handling Missing Values​
      
            Utilized linear regression to predict missing values in the target variable TAVG.​
      
      Interpolated missing values for essential weather parameters to create a more complete dataset.​
      
      Feature Scaling​
      
      Applied Min-Max Scaling to normalize feature values for better model performance.​
      
      Ensured uniform scaling across features to maintain consistency in the dataset.​

Data Cleaning​

Interpolated missing values for essential weather parameters to create a more complete dataset.​

Removed unnecessary columns such as 'NAME,' and 'PGTM,' during the preprocessing phase.​

Target Variable Transformation​

Transformed the target variable 'PRCP' to enhance prediction accuracy using appropriate scaling techniques.​

<h2>Data Splitting and Model Validation</h2>
Explain the data splitting strategy (e.g., 70/30 split) and the techniques employed for validating the models. Mention any specific considerations for each model.

<h2>API Integration</h2>
Detail the development of the API that integrates both models. Discuss the API endpoints, their purposes, and how users can leverage the API for weather predictions.

<h2>Usage</h2>
Provide instructions on how users can use the code, run the models, and interact with the API.

<h2>Contributing</h2>
Outline guidelines for contributors, including information on how to report issues, propose enhancements, and submit pull requests.

<h2>License</h2>
This project is part of an undergraduate computer science degree program and is not currently licensed for public use or distribution. All rights are reserved by the authors.
