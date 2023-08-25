# Plant-Disease-detection
Plant disease detection using deep learning is a powerful application of artificial intelligence that aims to automate the process of identifying and diagnosing diseases in plants. It involves using deep learning models, particularly convolutional neural networks (CNNs), to analyze images of plants and detect the presence of diseases or other health issues.
Data Preparation:

The script assumes there is a folder called "dataset/train" containing sub-folders for each plant disease category, with images of healthy plants and plants affected by various diseases.
The images are read, resized to a fixed size, and preprocessed to convert them to RGB format and then to HSV format.
Image Segmentation:

The script performs image segmentation to extract green and brown color regions from the original RGB image using appropriate HSV color thresholds. This step aims to separate the diseased regions from healthy regions in the plant images.
Feature Extraction:

Three types of global feature descriptors are used: Hu Moments, Haralick Texture, and Color Histogram.
Hu Moments describe the shape of an object and are computed from the grayscale segmented plant images.
Haralick Texture features characterize the texture of an image and are computed from the grayscale segmented plant images using the Mahotas library.
Color Histogram features represent the distribution of colors in the image and are computed from the segmented plant images in the HSV color space.
Data Labeling:

The script collects training labels from the sub-folder names in "dataset/train" and encodes them using LabelEncoder from scikit-learn.
Feature Scaling:

The extracted features are normalized using MinMaxScaler from scikit-learn to scale them in the range of [0, 1].
Saving Feature Data:

The script saves the preprocessed and scaled feature data along with the corresponding labels in two separate HDF5 files.
Model Training and Evaluation:

The script trains various machine learning models, including Logistic Regression, Linear Discriminant Analysis, K-Nearest Neighbors, Decision Trees, Random Forests, Naive Bayes, and Support Vector Machines, using k-fold cross-validation (10-fold).
![image](https://github.com/PranavBhanot/Plant-Disease-detection/assets/74693658/8c5714d6-c60b-4802-9be6-afe952402bb8)


The accuracy of each model is evaluated using confusion matrices and classification reports and was found out to be a staggering 97%.
The best-performing model (Random Forest) is trained on the entire dataset and used for making predictions on unseen data.
Saving Model:

The trained Random Forest model is saved using joblib for later use.
Test Data:

The script assumes there is a folder called "dataset/test" containing test images of plants.
Model Prediction:

The saved Random Forest model is loaded, and predictions are made on the test data.
The confusion matrix, classification report, and accuracy score are computed to evaluate the model's performance on the test set.
![image](https://github.com/PranavBhanot/Plant-Disease-detection/assets/74693658/ec37e1f4-a225-4461-a067-0d386aafb72e)

