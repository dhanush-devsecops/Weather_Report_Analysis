import pandas as pd

def clean_weather_data(file_path):
    # Load dataset
    df = pd.read_csv(file_path)
    
    # Clean column names (removes hidden spaces)
    df.columns = df.columns.str.strip()
    
    # Rename Kaggle columns to standard names
    rename_map = {
        'District': 'Area',
        'Max Temp (°C)': 'Temperature',
        'Rain (mm)': 'Rainfall',
        'Max Humidity (%)': 'Humidity',
        'Max Wind Speed (Kmph)': 'WindSpeed'
    }
    df = df.rename(columns=rename_map)
    
    # 1. Convert Date format correctly
    # 'coerce' turns bad dates into NaT (Not a Time) so the code doesn't crash
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    
    # 2. CRITICAL: Extract Month and Year for the Charts
    # This fixes the 'Month' KeyError you were seeing
    df['Month'] = df['Date'].dt.month
    df['Year'] = df['Date'].dt.year
    
    # 3. Handle Missing Values
    # We only apply mean imputation to numeric columns
    numeric_cols = df.select_dtypes(include=['number']).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
    
    # 4. Drop any rows where Date conversion failed
    df = df.dropna(subset=['Date'])
    
    return df