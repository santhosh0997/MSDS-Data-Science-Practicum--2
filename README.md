# Crop Yield Forecasting.


This repository contains the complete workflow for a **8-week machine learning project**, including:

- Data preprocessing  
- EDA  
- Feature engineering  
- Model building & hyperparameter tuning  
- Saving trained models  
- Interactive dashboard  

A `requirements.txt` file is included for easy environment setup.

---

## Setting Up the Environment

To avoid package conflicts, create a **virtual environment**.

### macOS / Linux
```bash
python3 -m venv venv
source venv/bin/activate
```

### Windows
```bash
python -m venv venv
venv\Scripts\activate
```

---

## Install Dependencies

With the virtual environment activated:

```bash
pip install -r requirements.txt
```

---

## Run the Streamlit Dashboard

Once the environment is active and all dependencies are installed, you can launch the application.

```bash
streamlit run dashboard.py
