## End to End Machine Learning Project

### Problem Statement

The Laptop Price Prediction project aims to develop a machine learning model to estimate the price of a laptop based on its features, such as brand, processor type, RAM, storage, screen size, and other specifications. With the wide variety of configurations available in the market, determining a fair price is challenging for consumers and businesses. This project addresses the problem by analyzing patterns in historical data and predicting accurate laptop prices, enabling informed decision-making for buyers and sellers


![Laptop_Price_Prediction](https://github.com/user-attachments/assets/928cff84-f92e-4c02-8442-3158e7f6d64d)

#### Table of Contents 

- [Introduction](#introduction)
- [Datasets](#Dataset)
- [Features](#features)
- [Getting Started](#getting-started)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Technologies Used](#technologies-used)

- ### Introduction 🌟

Laptop Price Prediction is a machine learning project that focuses on estimating the price of laptops based on their specifications. With laptops offering diverse configurations—ranging from basic models for everyday use to high-         performance devices for gaming or professional tasks—it can be challenging to assess their market value.By analyzing features such as brand, processor type, RAM, storage capacity, screen size, and additional attributes, this project    leverages data-driven insights to build a predictive model. The goal is to help consumers, retailers, and manufacturers make better decisions regarding pricing and purchasing, demonstrating the practical application of machine learning in real-world scenarios

- ### Datasets 

Dataset is taken from Kaggle and stored in  MYSQL Database

### Features 🚀

### Data Ingestion and Transformation 📊

We've introduced robust data ingestion and transformation components to preprocess raw data efficiently. This ensures data quality and reliability in our predictive models.

![laptop_price_prediction (2)](https://github.com/user-attachments/assets/fb6fe18a-cd8a-493b-9cad-288ceeddb527)

#### Web Application 🌐

We are thrilled to present our web-based user interface for easy input and prediction of Laptop Price. You can now interact with our model through a user-friendly web application.

#### Attributes in the Dataset:
- `Company` : Company Name
- `TypeName` : Type of Laptop you want (Notebook,ultrbook etc...)
- `OpSys` : Operating System Name
- `Ram` : Type your Preferred Ram
- `Weight` : Weight you want
- `Touchscreen`: If you want Touchscreen or not (1 for Toucherscreen 0 for not)
- `Ips`: If you want Ips panel or not(1 for yes 0 for no)
- `Ppi`: write Pixel Resolution (Ppi- Pixel per unit)
- `Cpu Brand`: Mention Cpu Brand (Intel core I5, Intel Core I7 etc....)
- `Gpu Brand` : Mention Gpu Brand (Intel,Nvidia,AMD etc...)
- `SSD` : How much SSD you want in your Laptop (Write in GB)
- `HDD` : How much HDD you want in your Laptop (Write in GB)
### Tech stack used
1. Python
2. FastAPI
3. MySQL
4. Machine Learning Algorithm : Linear Regression,Ridge and Lasso,Decision Tree,K-Neighbors Regressor,Random Forest Regressor.

### How to run project

#### Step 1: Clone the project

```bash
git clone https://github.com/amitrai740/LAPTOPPRICEPREDICTION.git
```

#### Step 2 : Create a conda environment after opening the repository

```bash
conda create -p venv python==3.8 -y
```
#### Step 3 : Install the requirements
```bash
pip install -r requirements.txt
```
#### Step 4:  Run the application server  
```
python app.py
```
### Step 5: 

1. Visit the web app. :- http://127.0.0.1:5000/predictdata
2. Enter the attributes of the Laptop in the input form.
3. Click the "Predict" button.
4. Receive the predicted price of the Laptop.

### Step 6 : Results on new Data

![Screenshot 2024-12-16 141838](https://github.com/user-attachments/assets/e07ab404-5710-40c7-86cd-2c7fb2721d52)

         
