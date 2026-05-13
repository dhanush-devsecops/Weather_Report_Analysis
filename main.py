import os
from src.data_cleaning import clean_weather_data
from src.analysis_viz import generate_visuals

def main():
    # Ensure outputs directory exists
    if not os.path.exists('outputs'):
        os.makedirs('outputs')

    data_path = 'data/Telangana_combined_weather_data.csv'

    print("--- Starting Weather Data Pipeline ---")
    
    # Step 1: Cleaning
    print("Cleaning data and handling missing values...")
    df = clean_weather_data(data_path)
    
    # Step 2: Insights
    peak_temp = df['Temperature'].max()
    max_rain = df['Rainfall'].max()
    total_rain = df['Rainfall'].sum()
    
    print(f"Analysis Complete:")
    print(f" - Peak Temperature: {peak_temp}°C")
    print(f" - Max Single-Day Rain: {max_rain} mm")
    print(f" - Total Rainfall in Period: {total_rain} mm")

    # Step 3: Visualization
    print("Generating Heatmap and Trend charts in /outputs...")
    generate_visuals(df)
    
    print("Done! Check your 'outputs' folder for results.")

if __name__ == "__main__":
    main()