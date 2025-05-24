# Suicidal Post Detection System

A machine learning-based system for detecting potential suicidal content in text, with a focus on providing immediate resources and support.

## Problem Statement

Suicide is a critical public health issue, with social media often being a platform where individuals express their distress. Early detection of suicidal ideation in online content can help in timely intervention and potentially save lives. This project aims to:

- Detect potential suicidal content in text
- Provide immediate access to mental health resources
- Offer a user-friendly interface for content analysis
- Support multiple input methods (direct text, file upload)

## Technical Approach

The system is built using a modern microservices architecture:

### Backend (FastAPI)
- RESTful API for model serving
- Input validation using Pydantic
- Error handling and logging
- Scalable model deployment

### Frontend (Streamlit)
- Interactive web interface
- Real-time predictions
- Visual feedback with probability distributions
- File upload support
- Language selection (currently English)

## Model Architecture

The system uses a deep learning model with the following specifications:

- **Model Type**: LSTM with GloVe embeddings
- **Input**: Text sequences (max length: 50 tokens)
- **Output**: Binary classification (suicidal/non-suicidal)



## Dataset Details

- **Source**: [Kaggle - Suicide Detection from Reddit Posts](https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch)
- **Size**: ~20,000 Reddit posts
- **Classes**:
  - `Suicidal`
  - `Non-Suicidal`


## Installation

1. Clone the repository:
    ```bash
git clone [repository-url]
cd suicidal-depression-detection
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Running the Application

### Local Development

1. Start the FastAPI backend:
```bash
python api.py
```
The API will be available at `http://localhost:8000`

2. Start the Streamlit frontend:
    ```bash
    streamlit run app.py
    ```
The web interface will be available at `http://localhost:8501`

### API Documentation

FastAPI provides automatic API documentation:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Usage

1. Access the web interface at `http://localhost:8501`
2. Choose your input method:
   - Direct text input
   - File upload (.txt files)
3. Select the language (currently English)
4. Click "Predict" to analyze the content
5. View the results:
   - Binary classification (Suicidal/Non-suicidal)
   - Confidence scores
   - Probability distribution

## Screenshots

[To be added: Screenshots of the application interface]

## Model Performance

[To be added: Model performance metrics and visualizations]

## Mental Health Resources

The application provides immediate access to mental health resources, particularly for users in India:

- **Vandrevala Foundation**: +91 9999 666 555 (24x7, Call/WhatsApp)
- **AASRA**: 09820466726 (24x7, Hindi/English)
- **KIRAN Mental Health Helpline**: 18005990019 (24x7)
- **Jeevan Aastha Helpline**: 1800 233 3330 (24x7)
- **Fortis Stress Helpline**: +91-8376804102 (24x7)

## Disclaimer

This tool is for informational purposes only and should not be considered a substitute for professional medical or mental health advice. If you or someone you know is in crisis, please seek help immediately.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
