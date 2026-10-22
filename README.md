# 🎓 EduTrack AI — Student Performance Prediction

EduTrack AI is a Python-based machine learning project built with **Streamlit**. It provides an interactive interface for exploring student data and predicting student performance using a trained machine learning model.

The project demonstrates how Python, data analysis, and machine learning can be combined to build an accessible web application for student performance analysis.

## ✨ Project Overview

EduTrack AI brings together student data, a trained prediction model, and a Streamlit interface in one application.

### Key Components

- **Student Data:** Uses a CSV dataset containing student training data.
- **Machine Learning:** Uses a trained model for student performance prediction.
- **Interactive Interface:** Built with Streamlit.
- **Data Processing:** Uses pandas to work with structured data.
- **Model Storage:** Loads a saved model from a Joblib file.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Streamlit | Interactive web application |
| pandas | Data handling and processing |
| Machine Learning | Student performance prediction |
| Joblib | Loading the saved model |

## 📁 Project Structure

```text
EduTrack-AI/
│
├── app.py
├── train_model.py
├── requirements.txt
├── .gitignore
│
├── data/
│   └── student_training_data.csv
│
└── model/
    └── performance_model.joblib
```

### File Description

- `app.py` — Main Streamlit application.
- `train_model.py` — Script for training the machine learning model.
- `requirements.txt` — Python dependencies required by the project.
- `data/student_training_data.csv` — Student training dataset.
- `model/performance_model.joblib` — Saved machine learning model.
- `.gitignore` — Specifies files and folders Git should ignore.

## ⚙️ Getting Started

Follow these steps to run EduTrack AI on your computer.

### 1. Clone the Repository

Open a terminal and run:

```bash
git clone https://github.com/Vrushali-DS/EduTrack-AI.git
```

### 2. Open the Project Folder

```bash
cd EduTrack-AI
```

### 3. Create a Virtual Environment (Recommended)

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can use Command Prompt instead:

```cmd
.venv\Scripts\activate.bat
```

### 4. Install Dependencies

```bash
python -m pip install -r requirements.txt
```

### 5. Run the Application

```bash
streamlit run app.py
```

Streamlit will provide a local URL, usually:

```text
http://localhost:8501
```

Open that URL in your browser to access the application.

## 🧠 Machine Learning Model

The project includes a saved model at:

```text
model/performance_model.joblib
```

The application uses this model for prediction. The training script is available in `train_model.py`.

To retrain the model, first review the training script and ensure the dataset and dependencies are available. Then run:

```bash
python train_model.py
```

**Note:** Retraining may replace the existing saved model, depending on the script's implementation.

## 📊 Dataset

The project uses:

`data/student_training_data.csv`

The dataset supplies the student training data used by the project. Please consult the CSV columns and `train_model.py` to understand the specific input features, target variable, and data preparation process.

## 🎯 Project Objectives

- Apply machine learning to a student performance prediction task.
- Work with structured data using pandas.
- Connect a saved model to a Streamlit application.
- Practice integrating Python, data processing, and machine learning.
- Build a practical project for a data science portfolio.

## 🚀 Future Improvements

Potential improvements, depending on the project requirements, include:

- Add model evaluation metrics and visualizations.
- Improve the presentation of prediction results.
- Add additional data exploration options.
- Document the dataset features and model evaluation results.
- Deploy the application online.

These are possible future enhancements, not claims about existing functionality.

## 👩
