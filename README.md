<h1>
      AI Precipitation Prediction Project
</h1>

![PixelBrella Main](precipitation_prediction_model-main/static/images/main.png)
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
       <li>Contributing</li>
       <li>License</li>
</ul>
</body>
<h2>Introduction</h2>
<body>Our project focuses on building a sophisticated weather prediction and analysis system that integrates machine learning algorithms, real-time weather data retrieval, and data analysis techniques to forecast and analyze weather conditions accurately. Rainfall is a critical component of our ecosystem. Rainfall influences agriculture, water resource management, urban planning, and disaster preparedness. The ability to accurately predict rainfall helps with daily planning as well as long-term sustainability. For this project specifically, we wanted to create a model to predict rainfall but also make a recommendation on whether to bring an umbrella or not based on the rainfall prediction.
<br />
<br />
      
Rainfall patterns can vary significantly making it challenging to accurately predict. Changes in weather and precipitation are factors that contribute to the difficulties in predicting rainfall. Our goal with this project is to provide an accurate way to predict rainfall. This model will use historical weather data collected over the past 20 years and uses methods to preprocess this data. Machine learning algorithms are employed to train the model. The model then learns patterns and relationships between various weather parameters and rainfall levels. Current weather data is then inputted into the model which allows it to make predictions about future rainfall.​</body>
<h2>Models</h2>
<h3>Daily Model</h3>
The first dataset we found was from the National Center for Environmental Information. For our model, we looked at data from the past 20 years. This is the dataset we originally worked with but after some experimentation, we found that it was limited in the weather data it offered.​


<a href="https://www.ncdc.noaa.gov/cdo-web/search">This is the link to the daily dataset.</a>

<h3>Hourly Model</h3>
For the hourly mode, we used a data set that is from Iowa State University. This dataset offered data every hour and we collected data from the past 10 years​. This dataset allowed us to select more weather data to make our model more robust.

<a href="https://mesonet.agron.iastate.edu/request/download.phtml">This is the link for the hourly dataset. </a>

<h2>Preprocessing</h2>
<h3>Daily Model</h3>
<h5>Data Cleaning</h5>
<ul>
    <li>Utilized imputation techniques, including linear regression, to fill missing values in 'TAVG,' 'TMIN,' 'TMAX,' 'WSF5,' 'SNOW,' and 'SNWD.'</li>
    <li>Interpolated missing values in the target variable 'SNOW' and 'SNWD' to enhance dataset completeness.</li>
</ul>
<h5>Column Removal</h5>
<ul>
    <li>Removed unnecessary columns such as 'weather_info,' 'weather_description,' and 'city_name' during data preparation.</li>
    <li>Streamlined the dataset to include only relevant features for precipitation prediction.</li>
</ul>
<h3>Hourly Model</h3>
<h5>Data Cleaning</h5>
<ul>
    <li>Addressed missing values in key weather parameters, including 'Temp (F),' 'Dew Point,' 'Humidity,' 'Wind Direction,' 'Wind Speed,' 'Sea Level Pressure,' 'Visibility,' and 'Hourly Prcp.'</li>
    <li>Implemented linear regression to impute missing values for selected weather parameters.</li>
    <li>Interpolated missing values in the target variable 'Hourly Prcp' to enhance dataset completeness.</li>
</ul>
<h5>Column Removal</h5>
<ul>
    <li>Removed unnecessary columns, including 'station' and 'Date,' during the data preparation phase.</li>
    <li>Streamlined the dataset to include only relevant features for precipitation prediction.</li>
</ul>
<h2>Data Splitting and Model Validation</h2>
<body>
     <h3>Daily Model</h3>
<h5>Data Splitting</h5>
<ul>
    <li>Split the daily dataset into training and testing sets using a 70/30 split ratio.</li>
</ul>
<h5>Model Validation</h5>
<ul>
    <li>Applied Min-Max scaling for large datasets, enhancing model performance.</li>
    <li>Utilized KNeighborsRegressor for training the daily model with 14 neighbors for robust predictions.</li>
    <li>Evaluated daily model performance using mean squared error (MSE) and R-squared metrics.</li>
</ul>
<h3>Hourly Model</h3>
<h5>Data Splitting</h5>
<ul>
    <li>Split the hourly dataset into training and testing sets using a 70/30 split ratio.</li>
</ul>
<h5>Model Validation</h5>
<ul>
    <li>Applied z-score normalization for large hourly datasets, enhancing model performance.</li>
    <li>Utilized RandomForestRegressor for training the hourly model, employing 100 decision trees for robust predictions.</li>
    <li>Evaluated hourly model performance using mean squared error (MSE) and R-squared metrics.</li>
</ul>
</body>
<h2>API Integration</h2>
<h3>OpenWeatherMap API</h3>
<body>
      <ul>
            <li>Used for obtaining current weather data and short-term weather forecasts.</li>​
            <li>Provided information such as temperature, humidity, wind speed, visibility, and precipitation.</li>
      </ul>
</body>
<h3>Tomorrow.io API (formerly ClimaCell)</h3>
<body>
        <ul>
            <li>Utilized for fetching dew point data, which is an essential feature for my models.</li>​
            <li>Provided detailed timelines for various weather parameters, including dew point, at different intervals.</li>
      </ul>
</body>


<h2>Contributors</h2>
<p>This project was a collaborative effort by the following contributors:</p>

<ul>
   <li><strong>Laura Fonseca-Llorca</li>
</ul>

<h2>License</h2>
This project is part of an undergraduate computer science degree program and is not currently licensed for public use or distribution. All rights are reserved by the authors.
