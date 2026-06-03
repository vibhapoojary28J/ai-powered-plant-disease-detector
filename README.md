# AI Powered Plant Disease Detector

An AI-powered plant disease detection system that uses Machine Learning and Support Vector Machine (SVM) to identify plant leaf diseases from images and provide treatment recommendations. The system helps farmers and agricultural practitioners detect diseases early, reduce crop loss, and improve plant health management.

## Project Overview

Plant diseases significantly affect agricultural productivity and food security. Traditional disease diagnosis requires expert knowledge and manual inspection, which may not always be available. This project automates plant disease detection using image processing and machine learning techniques.

The system processes uploaded leaf images, extracts important visual features, classifies diseases using an SVM model, and generates disease-specific treatment recommendations.

## Features

* Plant Disease Detection from Leaf Images
* Image Preprocessing and Noise Reduction
* Feature Extraction using PCA
* SVM-Based Disease Classification
* Disease-Specific Treatment Recommendations
* User-Friendly Disease Prediction Interface
* Automated Crop Health Monitoring Support

## Workflow

1. Upload Leaf Image
2. Image Preprocessing
3. Feature Extraction using PCA
4. Disease Classification using SVM
5. Disease Prediction
6. Treatment Recommendation Generation
7. Display Results to User

## Technologies Used

* Python
* Machine Learning
* OpenCV
* NumPy
* Pandas
* Scikit-Learn
* PCA (Principal Component Analysis)
* Support Vector Machine (SVM)

## Dataset

PlantVillage Dataset

## Model Performance

| Model         | Accuracy (%) |
| ------------- | ------------ |
| Random Forest | 56.9         |
| KNN           | 43.5         |
| ANN           | 76.0         |
| SVM           | 83.2         |

The SVM model achieved the highest accuracy of **83.2%** and was selected for disease prediction and recommendation.


## Applications

* Smart Agriculture
* Crop Health Monitoring
* Early Disease Detection
* Precision Farming
* Agricultural Decision Support Systems

## Future Improvements

* Deep Learning Based CNN Models
* Mobile Application Deployment
* Real-Time Disease Detection
* Improved Accuracy with Larger Datasets
* Support for Additional Plant Species and Diseases

## Conclusion

The project demonstrates how machine learning can be used to automate plant disease diagnosis. By combining image preprocessing, PCA-based feature extraction, and SVM classification, the system provides accurate disease predictions along with useful treatment recommendations, supporting sustainable agricultural practices.
