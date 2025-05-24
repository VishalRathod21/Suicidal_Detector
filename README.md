# Suicidal Post Detection App

## Overview

This application is a Streamlit-based tool designed to analyze text content (posts) and predict the likelihood of them containing suicidal ideation. It serves as a preliminary screening tool.

**Important Disclaimer:**
This tool is **for informational purposes only** and should **not** be considered a substitute for professional medical or mental health advice. If you or someone you know is in crisis, please seek help immediately from qualified professionals or emergency services. The application's predictions are based on a machine learning model and may not always be accurate.

## How it Works

The application utilizes a pre-trained machine learning model (specifically, an LSTM model trained with GLoVe embeddings) to process input text. The text is tokenized and padded to a fixed sequence length before being fed into the model. The model outputs a probability score, which is then used to classify the post as either "Potential Suicide Post" or "Non Suicide Post". A bar chart visualizes the confidence level of the prediction.

## Setup

To run this application locally, follow these steps:

1.  **Ensure you have Python installed** (version 3.6 or higher recommended). You can download it from [python.org](https://www.python.org/).

2.  **Clone the repository** (if the code is hosted on a version control platform):

    ```bash
    git clone <repository_url>
    cd suicidal_post_detection_app
    ```
    *(Replace `<repository_url>` with the actual URL of your repository and `suicidal_post_detection_app` with your project directory name)*

3.  **Navigate to the project directory** in your terminal.

4.  **Install the required Python packages** by running the following command. This assumes you have a `requirements.txt` file generated with all dependencies.

    ```bash
    pip install -r requirements.txt
    ```
    *(If you don't have a `requirements.txt`, you can create one by running `pip freeze > requirements.txt` after installing the necessary libraries like `streamlit`, `tensorflow`, `pandas`, `plotly`, etc.)*

5.  **Obtain the trained machine learning model files.** You need `tokenizer.pkl` and `model.h5`. These files should be placed inside a directory named `models` at the root level of your project directory. The expected path is `./models/tokenizer.pkl` and `./models/model.h5`.
    *(Note: The model files are typically large and might not be included in the repository. Instructions on where to download them should be provided if they are not committed.)*

## Running the Application

1.  Open your terminal and navigate to the root directory of the project (where `app.py` is located).

2.  Run the Streamlit application using the command:

    ```bash
    streamlit run app.py
    ```

3.  The application will open in your default web browser, or provide a local URL (usually `http://localhost:8501`) that you can open.

## Project Structure

```
suicidal_post_detection_app/
├── app.py           # Main Streamlit application file
├── requirements.txt   # List of Python dependencies
└── models/          # Directory containing the ML model files
    ├── tokenizer.pkl  # Tokenizer object
    └── model.h5       # Trained Keras model
```

## Contributing

If you'd like to contribute to this project, please feel free to fork the repository and submit a pull request with your changes. For major changes, please open an issue first to discuss what you would like to change.

## License

*(Add license information here, e.g., MIT, Apache 2.0, etc.)*

## Contact

If you have any questions or feedback, feel free to reach out. 