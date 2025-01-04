# Healthcare Support System  

**A Decision Support System for Disease Prediction and Early Detection**  

## Overview  
The **Healthcare Support System** is an innovative decision support system designed to predict the likelihood of a user suffering from any of 32 diseases based on selected symptoms. By leveraging machine learning algorithms and data from the Mayo Clinic, this system aids in early detection and provides tailored guidance for improved healthcare outcomes.  

## Features  
- **Disease Prediction**: Utilizes classification-based algorithms to analyze 132 symptom inputs and predict associated diseases.  
- **Interactive Web Application**: Built with Streamlit, providing an intuitive and user-friendly interface for seamless interaction.  
- **Enhanced Accuracy**: Boosted medical diagnosis accuracy by 5% through advanced data processing and predictive modeling.  
- **Redirection for Treatment**: Offers disease-specific guidance and recommendations for further treatment.  

## Machine Learning Approach  
- **Algorithm Comparison**: Compared multiple machine learning algorithms including:  
  - Linear Regression  
  - Ridge Regression  
  - Lasso Regression  
  - Elastic Net  
  - Support Vector Regression (SVR)  
  - Random Forest  
- **Evaluation Metrics**: Models were evaluated using metrics such as Root Mean Square Error (RMSE) and Mean Absolute Percentage Error (MAPE).  
- **Model Selection**: Random Forest emerged as the most effective model based on performance metrics.  
- **Model Deployment**: The trained Random Forest model was serialized into a pickle file, enabling efficient reuse for multiple analyses without retraining.  

## Data Source  
- The dataset was sourced from the **Mayo Clinic**, providing comprehensive and reliable data for disease prediction and treatment recommendations.  

## Technology Stack  
- **Programming Language**: Python  
- **Frameworks & Libraries**: Streamlit, Scikit-learn, Pandas, Numpy  
- **Model Deployment**: Pickle  
- **Data Source**: Mayo Clinic  

## How It Works  
1. The user selects their symptoms from a list of 132 options.  
2. The system applies the trained Random Forest model to predict the likelihood of any associated disease.  
3. The interface provides a detailed prediction, along with potential treatments and redirection for specific diseases.  

## Key Highlights  
- Collaborated in a 4-person team to develop a robust disease prediction algorithm.  
- Integrated insights from the Mayo Clinic to enhance prediction and provide actionable healthcare advice.  
- Streamlined the user experience through an interactive interface built with Python and Streamlit.  

## Future Scope  
- Expanding the database to include more diseases and symptoms.  
- Incorporating additional machine learning models for improved prediction accuracy.  
- Adding multilingual support to make the system accessible globally.  

## Contributions  
This project was developed by a 4-member team, demonstrating collaboration, innovation, and technical expertise in machine learning and healthcare solutions.  
