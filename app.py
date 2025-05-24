import streamlit as st
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
import plotly.express as px
import pandas as pd

token_form = pickle.load(open(r'models\tokenizer.pkl', 'rb'))
model = load_model("models\model.h5")

if __name__ == '__main__':
    st.title('Suicidal Post Detection App ')
    st.write("This application uses a machine learning model to analyze text content and predict the likelihood of it being a potential suicidal post.")
    st.warning("Disclaimer: This tool is for informational purposes only and should not be considered a substitute for professional medical or mental health advice. If you or someone you know is in crisis, please seek help immediately.")

    # Input Section
    with st.container():
        st.subheader("Input the Post content below")
        sentence = st.text_input("Enter your post content here")
        predict_btt = st.button("Predict")

    st.markdown("--- # Prediction Results Section")
    # Prediction Results Section
    if predict_btt:
        with st.container():
            # Define the post
            st.write("Post: " +sentence)
            twt = [sentence]
            twt = token_form.texts_to_sequences(twt)
            twt = pad_sequences(twt, maxlen=50)

            # Predict the ideation
            prediction = model.predict(twt)[0][0]
            # Print the prediction
            st.subheader("Prediction Results:")
            if(prediction > 0.5):
                 st.error("Result: Potential Suicide Post") # Using st.error for emphasis
            else:
                st.success("Result: Non Suicide Post")
            
            st.write("Confidence Level:")
            class_label = ["Potential Suicide Post","Non Suicide Post"]
            prob_list = [prediction*100,100-prediction*100]
            prob_dict = {"Category":class_label,"Probability (%)":prob_list}
            df_prob = pd.DataFrame(prob_dict)
            fig = px.bar(df_prob, x='Category', y='Probability (%)', text='Probability (%)')
            fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
            model_option = "LSTM+GLove"
            
            # Improve probability explanation
            st.write(f"The {{model_option}} model analyzes your post. Here's the probability distribution:")

            # Update layout title
            fig.update_layout(
                title_text=f"{{model_option}} Model Prediction Probability Comparison",
                xaxis_title="Category",
                yaxis_title="Probability (%)"
            )
            
            st.plotly_chart(fig, use_container_width=True)

            # Repeat disclaimer
            st.warning("Disclaimer: This tool is for informational purposes only and should not be considered a substitute for professional medical or mental health advice. If you or someone you know is in crisis, please seek help immediately.")

    st.markdown("--- # Resources Section")
    # Add resources section
    with st.container():
        st.subheader("Need Immediate Help?")
        st.write("If you are in immediate danger or need to speak with someone right away, please contact one of the resources below:")

        st.markdown("### India Specific Resources:")
        st.markdown("-   **Vandrevala Foundation:** +91 9999 666 555 (24x7, Call/WhatsApp)")
        st.markdown("-   **AASRA:** 09820466726 (24x7, Hindi/English)")
        st.markdown("-   **KIRAN Mental Health Helpline (Govt. of India):** 18005990019 (24x7)")
        st.markdown("-   **Jeevan Aastha Helpline:** 1800 233 3330 (24x7)")
        st.markdown("-   **Fortis Stress Helpline:** +91-8376804102 (24x7, multiple languages)")
        st.markdown("For a more comprehensive list of helplines across India, you can visit resources like [indianhelpline.com](https://indianhelpline.com/suicide-helpline).")
        st.warning("Remember: This app is not a substitute for professional help. If you are experiencing a mental health crisis, please reach out to a qualified healthcare provider or emergency services.")

    