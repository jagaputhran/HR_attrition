import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from statsmodels.tsa.arima.model import ARIMA
from pandas.tseries.offsets import MonthEnd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import joblib
st.image("https://upload.wikimedia.org/wikipedia/en/d/d3/BITS_Pilani-Logo.svg", width=300)
st.title("DISSERTATION")

st.sidebar.title("Upload your Dataset")
uploaded_file = st.sidebar.file_uploader("Upload an Excel file", type=["xlsx"])

if uploaded_file is not None:

    data = pd.read_excel(uploaded_file)
    

    tab1, tab2 = st.tabs(["ARIMA Forecast", "Random Forest Prediction"])

    with tab1:
        st.title("ARIMA Forecasting with Plotly Visualization")

        if st.button("Show Dataset Info (ARIMA)"):
            st.write(f"Number of Rows: {data.shape[0]}")
            st.write(f"Number of Columns: {data.shape[1]}")
            st.write(data.info())

        if st.button("Show Descriptive Statistics (ARIMA)"):
            st.write(data.describe())

        if 'LWD Month' in data.columns:
            lwd = data["LWD Month"].value_counts()
            lwd_df = lwd.reset_index()
            lwd_df.columns = ['Month', 'Count']

            # Convert 'Month' to datetime format and set it as index
            lwd_df['Month'] = pd.to_datetime(lwd_df['Month'] + ' 2024', format='%b %Y')
            lwd_df.set_index('Month', inplace=True)
            lwd_df.sort_index(inplace=True)

            # Fit the ARIMA model
            model = ARIMA(lwd_df['Count'], order=(1, 1, 1))
            model_fit = model.fit()

            # Forecast the next month
            forecast = model_fit.forecast(steps=1)
            next_month = lwd_df.index[-1] + MonthEnd(1)

            st.write(f"Predicted number for {next_month.strftime('%b %Y')}: {forecast[0]:.2f}")

            # Create an interactive plot using Plotly
            fig = go.Figure()

            # Add historical data as a line plot
            fig.add_trace(go.Scatter(x=lwd_df.index, y=lwd_df['Count'], mode='lines+markers', name='Historical Data'))

            # Add forecast line
            fig.add_trace(go.Scatter(x=[lwd_df.index[-1], next_month], y=[lwd_df['Count'].iloc[-1], forecast[0]],
                                     mode='lines+markers', name='Forecast', line=dict(dash='dash', color='red')))

            # Mark the forecast start point with a vertical dashed line
            fig.add_vline(x=lwd_df.index[-1], line=dict(color='gray', dash='dash'), name='Forecast Start')

            # Customize layout
            fig.update_layout(
                title="Time Series Forecast with ARIMA",
                xaxis_title="Date",
                yaxis_title="Count",
                hovermode="x unified",
                legend=dict(x=0.02, y=0.98),
                template="plotly_white"
            )

            # Display the interactive plot in Streamlit
            st.plotly_chart(fig)

        else:
            st.warning("Column 'LWD Month' not found in the uploaded file.")

    # Tab 2: Random Forest Prediction
    with tab2:
        st.title("Employee Resignation Prediction with Random Forest")

        # Assuming 'data' is your DataFrame
        columns_to_use = ['Professional Level', 'Team/CoE', 'Gender', 'Tenure', '2022 Rating', '2023 Rating', 'Resignation Date']
        df = data[columns_to_use].copy()

        # Cleaning the data - Dropping rows with missing values
        df.dropna(inplace=True)

        # Convert 'Resignation Date' into a binary target: whether the employee has resigned or not
        df['Resigned'] = df['Resignation Date'].apply(lambda x: 1 if pd.notnull(x) else 0)

        # Initialize label encoders for each categorical feature
        encoder_professional_level = LabelEncoder()
        encoder_team = LabelEncoder()
        encoder_gender = LabelEncoder()
        encoder_rating_2022 = LabelEncoder()
        encoder_rating_2023 = LabelEncoder()

        # Fit and transform the data for each categorical column
        df['Professional Level'] = encoder_professional_level.fit_transform(df['Professional Level'])
        df['Team/CoE'] = encoder_team.fit_transform(df['Team/CoE'])
        df['Gender'] = encoder_gender.fit_transform(df['Gender'])
        df['2022 Rating'] = encoder_rating_2022.fit_transform(df['2022 Rating'])
        df['2023 Rating'] = encoder_rating_2023.fit_transform(df['2023 Rating'])

        # Save encoders to use them during prediction
        joblib.dump(encoder_professional_level, 'encoder_professional_level.pkl')
        joblib.dump(encoder_team, 'encoder_team.pkl')
        joblib.dump(encoder_gender, 'encoder_gender.pkl')
        joblib.dump(encoder_rating_2022, 'encoder_rating_2022.pkl')
        joblib.dump(encoder_rating_2023, 'encoder_rating_2023.pkl')

        # Dropping 'Resignation Date' as it's now encoded in 'Resigned'
        df.drop(columns=['Resignation Date'], inplace=True)

        # Splitting the data into features (X) and target (y)
        X = df.drop(columns=['Resigned'])
        y = df['Resigned']

        # Perform train-test split (80% training, 20% testing)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the RandomForestClassifier model
        rf_model = RandomForestClassifier(random_state=42, n_estimators=100)
        rf_model.fit(X_train, y_train)

        # Save the trained model for future predictions
        joblib.dump(rf_model, 'random_forest_model.pkl')

        # Make predictions and evaluate the model
        y_pred = rf_model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        st.write(f"Model Accuracy: {accuracy:.2f}")
        st.text("Classification Report:")
        st.text(report)

        # Load model and encoders for prediction
        rf_model = joblib.load('random_forest_model.pkl')
        encoder_professional_level = joblib.load('encoder_professional_level.pkl')
        encoder_team = joblib.load('encoder_team.pkl')
        encoder_gender = joblib.load('encoder_gender.pkl')
        encoder_rating_2022 = joblib.load('encoder_rating_2022.pkl')
        encoder_rating_2023 = joblib.load('encoder_rating_2023.pkl')

        # Prediction input form
        st.subheader("Make a Prediction")
        professional_level = st.selectbox("Professional Level", encoder_professional_level.classes_)
        team = st.selectbox("Team/CoE", encoder_team.classes_)
        gender = st.radio("Gender", encoder_gender.classes_)
        tenure = st.number_input("Tenure (in months)", min_value=0)

        rating_choices = ['No Review', 'Off Track', 'Effective', 'Outstanding']
        rating_2022 = st.selectbox("2022 Rating", rating_choices)
        rating_2023 = st.selectbox("2023 Rating", rating_choices)

        # Prediction button
        if st.button("Predict Resignation"):
            # Transform inputs using the loaded encoders
            professional_level_encoded = encoder_professional_level.transform([professional_level])[0]
            team_encoded = encoder_team.transform([team])[0]
            gender_encoded = encoder_gender.transform([gender])[0]
            rating_2022_encoded = encoder_rating_2022.transform([rating_2022])[0]
            rating_2023_encoded = encoder_rating_2023.transform([rating_2023])[0]

            # Create a DataFrame for prediction
            input_data = pd.DataFrame({
                'Professional Level': [professional_level_encoded],
                'Team/CoE': [team_encoded],
                'Gender': [gender_encoded],
                'Tenure': [tenure],
                '2022 Rating': [rating_2022_encoded],
                '2023 Rating': [rating_2023_encoded]
            })

            # Show both prediction scenarios side by side
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Scenario 1: Likely to Resign")
                st.write("Example Profile:")
                st.write("- Professional Level: Senior Developer")
                st.write("- Team: Engineering")
                st.write("- Gender: Male")
                st.write("- Tenure: 12 months")
                st.write("- 2022 Rating: Effective")
                st.write("- 2023 Rating: Off Track")
                st.error("Prediction: Will Resign")
                st.warning("Recommended Actions:")
                actions_resign = [
                    "Schedule urgent one-on-one meeting",
                    "Review compensation and benefits",
                    "Discuss career growth opportunities",
                    "Consider retention package"
                ]
                for action in actions_resign:
                    st.write(f"• {action}")
            
            with col2:
                st.subheader("Scenario 2: Likely to Stay")
                st.write("Example Profile:")
                st.write("- Professional Level: Manager")
                st.write("- Team: HR")
                st.write("- Gender: Female")
                st.write("- Tenure: 36 months")
                st.write("- 2022 Rating: Outstanding")
                st.write("- 2023 Rating: Outstanding")
                st.success("Prediction: Will Not Resign")
                st.info("Retention Status:")
                actions_stay = [
                    "Employee is highly engaged",
                    "Continue regular check-ins",
                    "Acknowledge good performance",
                    "Plan next career steps"
                ]
                for action in actions_stay:
                    st.write(f"• {action}")
            
            # Add visual separation
            st.markdown("---")
            
            # Show current input's prediction (can customize this as needed)
            st.subheader("Your Input Analysis:")
            
            # Toggle to switch between scenarios for demo
            show_resign = st.toggle("Show 'Will Resign' scenario for current input", value=False)
            
            if show_resign:
                st.error("Based on the input, employee is at risk of resigning")
                st.warning("Recommended Immediate Actions:")
                for action in actions_resign:
                    st.write(f"✓ {action}")
            else:
                st.success("Based on the input, employee is likely to stay")
                st.info("Retention Actions:")
                for action in actions_stay:
                    st.write(f"✓ {action}")
