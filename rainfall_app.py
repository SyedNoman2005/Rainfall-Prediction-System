import sys

# guard imports to surface environment problems early
try:
    import streamlit as st
    import pandas as pd
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import mean_squared_error, r2_score
except ImportError as exc:
    # Exit with clear instructions if a library cannot be loaded (often numpy)
    sys.exit(
        "\n" +
        "🚨 Failed to import required libraries. "
        "This usually means your Python environment has incompatible packages.\n" +
        f"Original error: {exc}\n" +
        "Please reinstall numpy/pandas or use a Python version supported by your dependencies.\n"
    )

# Set page title
st.set_page_config(page_title="Rainfall Predictor", page_icon="🌧️")
st.title("🌧️ Rainfall Prediction System")
st.write("Enter weather conditions to predict rainfall amount")

# Step 3: Create synthetic dataset
@st.cache_data
def create_dataset():
    np.random.seed(42)
    n_samples = 1000
    
    states = ['Kerala', 'Maharashtra', 'Tamil Nadu', 'Karnataka', 'Rajasthan']
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    
    data = {
        'State': np.random.choice(states, n_samples),
        'Month': np.random.choice(months, n_samples),
        'Temperature': np.random.uniform(15, 40, n_samples),  # Celsius
        'Humidity': np.random.uniform(30, 95, n_samples),     # Percentage
        'Wind_Speed': np.random.uniform(0, 30, n_samples),    # km/h
        'Pressure': np.random.uniform(990, 1020, n_samples),  # hPa
    }
    
    # Create rainfall based on conditions
    data['Rainfall'] = (
        (data['Humidity'] * 0.8) + 
        (data['Wind_Speed'] * 0.5) + 
        (100 - data['Temperature']) + 
        (data['Pressure'] - 1000) * 0.3 +
        np.random.normal(0, 20, n_samples)  # Add noise
    )
    
    # Ensure no negative rainfall
    data['Rainfall'] = np.maximum(0, data['Rainfall'])
    
    return pd.DataFrame(data)

# Load dataset
df = create_dataset()

# Show data preview
if st.checkbox("Show raw data"):
    st.subheader("Sample of Training Data")
    st.write(df.head(10))

# Step 4: Preprocess data
def preprocess_data(df, user_input=None):
    # One-hot encoding for categorical variables
    df_encoded = pd.get_dummies(df, columns=['State', 'Month'])
    
    if user_input is not None:
        # Create dataframe from user input
        user_df = pd.DataFrame([user_input])
        user_encoded = pd.get_dummies(user_df, columns=['State', 'Month'])
        
        # Align columns with training data
        train_columns = df_encoded.drop('Rainfall', axis=1).columns
        user_encoded = user_encoded.reindex(columns=train_columns, fill_value=0)
        
        return df_encoded, user_encoded
    
    return df_encoded

# Prepare training data
df_encoded = preprocess_data(df)

# Separate features and target
X = df_encoded.drop('Rainfall', axis=1)
y = df_encoded['Rainfall']

# Step 5: Split and train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Model evaluation
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.sidebar.subheader("📊 Model Performance")
st.sidebar.write(f"R² Score: {r2:.3f}")
st.sidebar.write(f"Mean Squared Error: {mse:.2f}")

# Step 6: Build user interface
st.subheader("📝 Enter Weather Conditions")

# Create input fields
col1, col2 = st.columns(2)

with col1:
    state = st.selectbox("Select State", ['Kerala', 'Maharashtra', 'Tamil Nadu', 'Karnataka', 'Rajasthan'])
    month = st.selectbox("Select Month", ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
    temperature = st.slider("Temperature (°C)", 15.0, 40.0, 28.0)

with col2:
    humidity = st.slider("Humidity (%)", 30.0, 95.0, 70.0)
    wind_speed = st.slider("Wind Speed (km/h)", 0.0, 30.0, 15.0)
    pressure = st.slider("Pressure (hPa)", 990.0, 1020.0, 1010.0)

# Step 7: Make prediction
if st.button("🌧️ Predict Rainfall", type="primary"):
    # Prepare user input
    user_input = {
        'State': state,
        'Month': month,
        'Temperature': temperature,
        'Humidity': humidity,
        'Wind_Speed': wind_speed,
        'Pressure': pressure
    }
    
    # Preprocess user input
    _, user_encoded = preprocess_data(df, user_input)
    
    # Make prediction
    prediction = model.predict(user_encoded)[0]
    
    # Display result
    st.subheader("📈 Prediction Result")
    
    # Create colorful result box
    if prediction < 10:
        color = "🟢"
        condition = "Low rainfall"
    elif prediction < 30:
        color = "🟡"
        condition = "Moderate rainfall"
    elif prediction < 60:
        color = "🟠"
        condition = "High rainfall"
    else:
        color = "🔴"
        condition = "Very high rainfall (Heavy rain expected!)"
    
    st.markdown(f"""
    <div style="padding: 20px; border-radius: 10px; background-color: #f0f2f6; text-align: center;">
        <h2 style="color: #333;">{color} {prediction:.1f} mm</h2>
        <p style="font-size: 18px;">{condition}</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Show input summary
    with st.expander("View input summary"):
        st.write(f"**State:** {state}")
        st.write(f"**Month:** {month}")
        st.write(f"**Temperature:** {temperature}°C")
        st.write(f"**Humidity:** {humidity}%")
        st.write(f"**Wind Speed:** {wind_speed} km/h")
        st.write(f"**Pressure:** {pressure} hPa")

# Add information section
with st.sidebar.expander("ℹ️ About this app"):
    st.write("""
    This app predicts rainfall using Linear Regression based on:
    - Location (State)
    - Time (Month)
    - Weather conditions (Temperature, Humidity, Wind Speed, Pressure)
    
    **Note:** This is a demonstration using synthetic data.
    """)

# Show feature importance
if st.sidebar.checkbox("Show Feature Importance"):
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'coefficient': model.coef_
    }).sort_values('coefficient', ascending=False)
    
    st.sidebar.bar_chart(feature_importance.set_index('feature'))