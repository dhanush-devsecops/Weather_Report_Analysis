import matplotlib.pyplot as plt
import seaborn as sns

def generate_visuals(df):
    # Set style
    sns.set_theme(style="whitegrid")
    
    # 1. Correlation Heatmap
    plt.figure(figsize=(10, 8))
    correlation_matrix = df.select_dtypes(include=['number']).corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('Weather Feature Correlation Heatmap')
    plt.savefig('outputs/correlation_heatmap.png')
    plt.close()
    
    # 2. Temperature Trend
    plt.figure(figsize=(12, 6))
    plt.plot(df['Date'], df['Temperature'], marker='o', color='orange', label='Temp (°C)')
    plt.title('Daily Temperature Trends (2023)')
    plt.xlabel('Date')
    plt.ylabel('Temperature (°C)')
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig('outputs/temp_trend.png')
    plt.close()

    # 3. Monthly Rainfall Bar Chart
    plt.figure(figsize=(10, 6))
    monthly_rain = df.groupby('Month')['Rainfall'].sum()
    monthly_rain.plot(kind='bar', color='skyblue')
    plt.title('Total Rainfall by Month')
    plt.ylabel('Rainfall (mm)')
    plt.savefig('outputs/rainfall_distribution.png')
    plt.close()