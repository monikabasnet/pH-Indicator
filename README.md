# 🧪 pH Predictor Web App

This is a Django-based web application that predicts the **pH value** from a litmus strip image using a **Random Forest** model trained on feature vectors extracted from images via **InceptionV3**.

## 💡 Features
- Upload an image of a litmus strip
- Extract image features using InceptionV3
- Predict the pH using a trained Random Forest model
- Clean UI for image upload and prediction display

## 🛠️ Tech Stack
- Python, Django
- TensorFlow, Scikit-learn, Joblib
- HTML, CSS (Django templates)
- Virtual Environment for isolation

## 🚀 How to Run Locally

```bash
# Clone the repo
git clone https://github.com/monikabasnet/pH-Indicator.git
cd pH-Indicator

# Create & activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run the server
python manage.py runserver
