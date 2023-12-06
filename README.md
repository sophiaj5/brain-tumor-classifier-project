# Capstone: Brain Tumor Classifier

## Problem Statement

Brain tumors represent a complex and diverse category of medical conditions, causing tremendous challenges for an accurate and timely diagnosis. The demanding nature of healthcare work, often characterized by long hours, inadequate nutrition, and minimal sleep, makes the process of a precise diagnoses that much more difficult. This project aims to develop a machine learning classification model to accurately distinguish between four types of brain tumors: gliomas, meningiomas, no tumors, and pituitary tumors.

## Data Dictionary

The dataset for this project can be found at https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset?rvi=1

|**glioma**| glioma folder in training and testing folders | scan showing a glioma tumor

|**meningioma**| meningioma folder in training and testing folders | scan showing a meningioma tumor

|**no tumor**| no tumor folder in training and testing folders | clean scan not showing any tumors

|**pituitary**| pituitary folder in training and testing folders | scan showing a pituitary tumor

## Executive Summary

### Preprocessing
There were a number of steps taken to preprocessing the images to prepare them for the modelling process.

The preprocessing began with the retrieval of data from Google Drive, where it was organized into training and testing folders. This dataset includes glioma, meningioma, no tumor, and pituitary folders, each with their respective images in all of the folders. The beginning of this process involved creating a function named `read_images` on both sets of folders.

Within the `read_images` function, the paths to the images were found. The images were given a specific target size then converted into arrays. The resulting images and their corresponding labels were converted to numpy arrays and then returned by the function.

Both the training and testing folders were shuffled and a train test split was performed. The y train and test variables (the labels) were dummified to convert the categorical values to numerical. The shape was found for both the X train and y train to ensure a match and finally, both the X train and X test variables were divided by 255.

## Models 
Four models were created in the process of trying to find the best one for this project.

1) `brain-model1` The first model was very simple and used one Conv2D and MaxPooling layer, and was Flattened before adding the last Dense layer with softmax activation.

2) `brain-model2` The second model was first flattened and then two Dense layers with relu activation were used before the final Dense layer with softmax activation. An Early Stop was also used to prevent overfitting.

3) `brain-model3` The third model was first flattened, then two Dense layers both with relu activation and l2 regularizers of 0.002 were used before the final Dense layer with softmax activation and l2 regularizer of 0.002.

4) `brain-model4` The final model used a number of Conv2D layers with relu activation and MaxPooling, a Dense layer, a Dropout of 0.5, and a final Dense layer with a softmax activation.

### Metrics
Various metrics were performed on all four models to compare how well the models performed.

#### Side Note
The metric of success chosen for this model was the f1 score because it balances precision and recall and handles imbalanced classes well.

#### Baseline Score
The baseline score is used to find the most basic model possible and how it will perform against the data we have. We can compare the rest of our metrics to the baseline score to get an idea of how well our model performed.

All of our models beat the baseline scores for this project.

#### Precision
The precision score was found on the test data against the predicted data for all of the models. The precision score tells you of all the positives that were predicted in the dataset, how many of them were actually positive.

1) Model 1 Scores:
- glioma: 0.93
- meningioma: 0.90
- no tumor: 0.98
- pituitary: 0.98

2) Model 2 Scores:
- glioma: 0.63
- meningioma: 0.93
- no tumor: 0.91
- pituitary: 0.93

3) Model 3 Scores:
- glioma: 0.66
- meningioma: 0.45
- no tumor: 0.93
- pituitary: 0.56

4) Model 4 Scores:
- glioma: 0.97
- meningioma: 0.93
- no tumor: 0.99
- pituitary: 0.99

#### Accuracy

1) Model 1 Score: 0.95

2) Model 2 Score: 0.82

3) Model 3 Score: 0.66

4) Model 4 Score: 0.97

#### Recall
The recall score was found on the testing data against the predicted data for all the models. This score tells you of all the positives in the data, how many the model correctly predicted.

1) Model 1 Scores:
- glioma: 0.91
- meningioma: 0.90
- no tumor: 1.00
- pituitary: 0.99

2) Model 2 Scores:
- glioma: 0.94
- meningioma: 0.38
- no tumor: 0.99
- pituitary: 0.93

3) Model 3 Scores:
- glioma: 0.74
- meningioma: 0.20
- no tumor: 0.72
- pituitary: 0.98

4) Model 4 Scores:
- glioma: 0.94
- meningioma: 0.94
- no tumor: 1.00
- pituitary: 0.99

#### f1 Score
The f1 score was found of the testing data against the predicted data for the all of the models. This score finds an overall score for your model by taking a balance between your precision and recall score.

1) Model 1 Scores:
- glioma: 0.92
- meningioma: 0.90
- no tumor: 0.99
- pituitary: 0.99

2) Model 2 Scores:
- glioma: 0.76
- meningioma: 0.54
- no tumor: 0.95
- pituitary: 0.93

3) Model 3 Scores:
- glioma: 0.70
- meningioma: 0.28
- no tumor: 0.81
- pituitary: 0.72

4) Model 4 Scores:
- glioma: 0.95
- meningioma: 0.93
- no tumor: 0.99
- pituitary: 0.99

#### Confusion Matrix
A confusion matrix shows the values of the True Positives, False Negatives, False Positives, and True Negatives. Confusion matrices were created for all four models and played a factor in deciding which model was the best for this project.

For this project, we looked specifically at False Positives and False Negatives and optimizing for them.

## Streamlit App
The Streamlit App gives a visual representation of the model at work. To run this Streamlit App, run `brain-model4.ipynb` and download the `model4.h5` file to your machine and move it to the `models` folder. Navigate to the streamlit folder through your command line and run the command: `streamlit run app.py`. This will open up the Streamlit Application in a browser on your local host.

From here, you are able to input an image of a brain scan and the model will let you know the type of tumor you most likely have, along with the percentage of accuracy of the diagnosis.

## Conclusions and Recommendations

In summary, the performance of all the models in distinguishing between different types of brain tumors was notable, with varying degrees of success. Among these models, Model 4 was the top performer across diverse metrics and graphical analyses. The performance of Model 4 was further confirmed through success on both the training and testing datasets, as demonstrated in the live Streamlit App.

Looking ahead, there are several recommendations and potential next steps for further exploration in this project. Firstly, enhancing the dataset by incorporating additional tumors and scans can introduce more diversity. We can also dive further into our model to find out exactly which images the model was incorrectly predicting which can offer a better idea on how to handle these cases. Additionally, extending the experimentation to create and evaluate more than four models, each with unique layers and configurations, would provide a comprehensive understanding of which architecture gives us optimal performance.