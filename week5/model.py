# import libraries
import boto3, re, sys, math, json, os, sagemaker, urllib.request
from sagemaker import get_execution_role
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image
from IPython.display import display
from time import gmtime, strftime
from sagemaker.serializers import CSVSerializer

# Define IAM role
role = get_execution_role()
prefix = 'sagemaker/DEMO-xgboost-dm'
my_region = boto3.session.Session().region_name # set the region of the instance

# this line automatically looks for the XGBoost image URI and builds an XGBoost container.
xgboost_container = sagemaker.image_uris.retrieve("xgboost", my_region, "latest")

print("Success - the MySageMakerInstance is in the " + my_region + " region. You will use the " + xgboost_container + " container for your SageMaker endpoint.")



# Define IAM role
bucket_name = 'week3-07312023' 
s3 = boto3.resource('s3')
try:
    if  my_region == 'us-east-1':
      s3.create_bucket(Bucket=bucket_name)
    else: 
      s3.create_bucket(Bucket=bucket_name, CreateBucketConfiguration={ 'LocationConstraint': my_region })
    print('S3 bucket created successfully')
except Exception as e:
    print('S3 error: ',e)


from sklearn.model_selection import train_test_split

# Load the Iris dataset
diabetes= pd.read_csv('diabetes_prediction_dataset.csv')

X= diabetes[['gender','age','hypertension','heart_disease','blood_glucose_level']]
y= diabetes['diabetes']
# Prepare the data
train_data, test_data = train_test_split(diabetes[['gender','age','hypertension','heart_disease','blood_glucose_level','diabetes']], test_size=0.2, random_state=42)
print(train_data.shape, test_data.shape)

# Save the data to CSV files
train_data.to_csv('train.csv', index=False, header=False)
test_data.to_csv('test.csv', index=False, header=False)

# Upload the data to S3
boto3.Session().resource('s3').Bucket(bucket_name).Object(os.path.join(prefix, 'train/train.csv')).upload_file('train.csv')
boto3.Session().resource('s3').Bucket(bucket_name).Object(os.path.join(prefix, 'test/test.csv')).upload_file('test.csv')

sess = sagemaker.Session()

# Set up SageMaker session and XGBoost Estimator
xgboost_container = sagemaker.image_uris.retrieve("xgboost", boto3.Session().region_name, "1.2-1")

xgb = sagemaker.estimator.Estimator(xgboost_container, role, instance_count=1, instance_type='ml.m4.xlarge',
                                    output_path='s3://{}/{}/output'.format(bucket_name, prefix), sagemaker_session=sess)

# Update objective and num_class for multiclass classification
xgb.set_hyperparameters(max_depth=5,eta=0.2,gamma=4,min_child_weight=6,subsample=0.8,silent=0,objective='binary:logistic',num_round=100)

# Fit the model on the training data
s3_input_train = sagemaker.inputs.TrainingInput(s3_data='s3://{}/{}/train'.format(bucket_name, prefix), content_type='csv')
xgb.fit({'train': s3_input_train})

# Deploy the model as an endpoint
xgb_predictor = xgb.deploy(initial_instance_count=1, instance_type='ml.m4.xlarge')

# Prepare test data for prediction
test_data_array = test_data.drop(['diabetes'], axis=1).values
xgb_predictor.serializer = CSVSerializer()  # Set the serializer type
predictions = xgb_predictor.predict(test_data_array).decode('utf-8')  # Predict
predictions_array = np.fromstring(predictions[1:], sep=',')  # Convert predictions to an array


# Calculate and print classification report and confusion matrix
from sklearn.metrics import classification_report, confusion_matrix

print("\nClassification Report:\n", classification_report(test_data['target'], np.round(predictions_array)))
print("\nConfusion Matrix:\n", confusion_matrix(test_data['target'], np.round(predictions_array)))

# Delete the endpoint after use
xgb_predictor.delete_endpoint()



