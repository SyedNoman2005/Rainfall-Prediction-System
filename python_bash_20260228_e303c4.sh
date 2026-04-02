# Create project folder
mkdir rainfall-predictor
cd rainfall-predictor

# Create virtual environment
python -m venv rainfall_env

# Activate it (Windows)
rainfall_env\Scripts\activate
# Or (Mac/Linux)
source rainfall_env/bin/activate

# Install required packages
pip install pandas scikit-learn streamlit numpy