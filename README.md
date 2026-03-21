# MLOps Project - Machine Failure Prediction

## Objective
This project demonstrates a simple MLOps workflow for machine failure prediction.

## Project Structure

mlops-project/
│
├── data/
│   ├── machine_failure_train.csv
│   └── machine_failure_predict.csv
│
├── notebooks/
│
├── src/
│   ├── train_model.py
│   └── predict_model.py
│
├── models/
│
├── app/
│
├── requirements.txt
├── README.md
└── .gitignore

## Workflow

### Step 1: Train the model
Run the following command:

python src/train_model.py

This will:
- load the training dataset
- split the data into train and test sets
- train the model
- evaluate accuracy
- save the trained model in the models folder

### Step 2: Run predictions
Run the following command:

python src/predict_model.py

This will:
- load the prediction dataset
- load the saved model
- generate predictions
- display results on screen
- save results as prediction_output.csv inside models folder

## Input Features
- temperature
- vibration
- pressure
- humidity
- runtime_hours
- maintenance_history

## Target Variable
- failure
  - 0 = Normal Operation
  - 1 = Failure Risk