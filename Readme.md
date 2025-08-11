Project Title: Smart Sorting - Fruit/Vegetable Classification (Healthy vs Rotten)

Description:
------------
Smart Sorting is a machine learning web application that classifies fruits and vegetables as either "Healthy" or "Rotten" based on the uploaded image. It uses a deep learning model (VGG16) trained on 28 classes (14 fruits/vegetables, each with a healthy and rotten version).

Technologies Used:
------------------
- Python
- Flask (Web Framework)
- TensorFlow / Keras (for model training and prediction)
- HTML & CSS (for frontend UI)
- VGG16 Pre-trained CNN model

Folder Structure:
-----------------
- app.py               → Flask backend logic
- templates/
    - index.html       → Image upload page
    - result.html      → Prediction result page
- static/
    - uploads/         → Stores uploaded images temporarily
- healthy_vs_rotten.h5 → Trained model file
- dataset/             → Training and testing image datasets

How to Run the Project:
-----------------------
1. Clone or download the project files.
2. Activate your virtual environment (if any).
3. Install the required packages using:
   pip install -r requirements.txt

4. Run the Flask application:
   python app.py

5. Open a browser and visit:
   http://127.0.0.1:5000

6. Upload an image of a fruit or vegetable to get the classification result.

Classes:
--------
- Apple__Healthy / Apple__Rotten
- Banana__Healthy / Banana__Rotten
- Bell_pepper__Healthy / Bell_pepper__Rotten
- Carrot__Healthy / Carrot__Rotten
- Cucumber__Healthy / Cucumber__Rotten
- Grape__Healthy / Grape__Rotten
- Guava__Healthy / Guava__Rotten
- Jujube__Healthy / Jujube__Rotten
- Mango__Healthy / Mango__Rotten
- Orange__Healthy / Orange__Rotten
- Pomegranate__Healthy / Pomegranate__Rotten
- Strawberry__Healthy / Strawberry__Rotten
- Tomato__Healthy / Tomato__Rotten

Author:
-------
Bharghavi

Notes:
------
- Make sure the `static/uploads` folder exists before running.
- The model file (`healthy_vs_rotten.h5`) must be present in the root directory.
- Uploaded files are stored temporarily and can be auto-cleared if needed.

Data Analysis Tools:
--------------------
These optional utility scripts are provided to inspect and visualize the dataset:
- split_dataset.py
- train_model.py
- class_distribution.py
- visualize_image.py
- visualize_grid.py
