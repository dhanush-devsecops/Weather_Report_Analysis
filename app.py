import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from src.data_cleaning import clean_weather_data

# Page Configuration
st.set_page_config(page_title="Advanced Weather Analytics | CODTECH", layout="wide")

st.title("🌦️ Advanced Weather Intelligence Dashboard")
st.markdown("### Data Science Internship - CODTECH IT Solutions (2025)")

try:
    # 1. LOAD AND CLEAN DATA
    data_path = 'data/Telangana_combined_weather_data.csv'
    df_raw = clean_weather_data(data_path)

    # 2. SIDEBAR FILTERS (Search Area)
    st.sidebar.header("🔍 Exploration Settings")
    if 'Area' in df_raw.columns:
        areas = sorted(df_raw['Area'].unique())
        selected_area = st.sidebar.selectbox("🔍 Search Area/District", areas)
        # Filter for the selected area (taking 35 records for analysis)
        df = df_raw[df_raw['Area'] == selected_area].head(35)
        st.subheader(f"Showing Results for: {selected_area}")
    else:
        df = df_raw.head(35)
        st.sidebar.warning("Area column not found in CSV.")

    # 3. KPI METRICS
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Peak Temp", f"{df['Temperature'].max()}°C")
    m2.metric("Total Rainfall", f"{df['Rainfall'].sum():.1f} mm")
    m3.metric("Avg Humidity", f"{df['Humidity'].mean():.1f}%")
    m4.metric("Max Wind", f"{df['WindSpeed'].max()} km/h")

    st.divider()

    # 4. MACHINE LEARNING SECTION
    st.subheader("🤖 AI Rain Prediction Model (Machine Learning)")
    # Prepare ML model using the full dataset for better accuracy
    ml_df = df_raw.dropna(subset=['Temperature', 'Humidity', 'WindSpeed', 'Rainfall'])
    X = ml_df[['Temperature', 'Humidity', 'WindSpeed']]
    y = (ml_df['Rainfall'] > 0).astype(int)

    if len(ml_df) > 5:
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)

        c_input, c_viz = st.columns([1, 2])
        with c_input:
            st.write("Predict rain probability based on inputs:")
            in_temp = st.number_input("Temperature (°C)", value=float(df['Temperature'].mean()))
            in_hum = st.slider("Humidity (%)", 0, 100, 50)
            in_wind = st.slider("Wind Speed (km/h)", 0, 100, 15)
            
            if st.button("Run AI Prediction"):
                pred = model.predict([[in_temp, in_hum, in_wind]])
                if pred[0] == 1:
                    st.error("🌧️ Prediction: Rain Likely")
                else:
                    st.success("☀️ Prediction: No Rain Likely")
        with c_viz:
            importances = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_})
            fig_ml = px.bar(importances, x='Importance', y='Feature', orientation='h', title="ML Feature Importance")
            st.plotly_chart(fig_ml, use_container_width=True)

    st.divider()

    # 5. DASHBOARD TABS (Matches your Report Figures)
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Interactive Trends", 
        "🌧️ Rainfall Distribution", 
        "📅 Monthly Averages",
        "🔗 Correlation Heatmap"
    ])

    with tab1:
        st.subheader("Temperature Trend Analysis")
        fig_trend = px.line(df, x='Date', y='Temperature', title=f"Temperature Over Time: {selected_area}",
                            hover_data=['Rainfall', 'Humidity'], markers=True)
        fig_trend.update_traces(line_color='#FF5733')
        st.plotly_chart(fig_trend, use_container_width=True)

    with tab2:
        st.subheader("Rainfall Distribution (Blue Bar Chart)")
        col_rain_chart, col_rain_txt = st.columns([2, 1])
        with col_rain_chart:
            fig_rain = px.bar(df, x='Date', y='Rainfall', color_discrete_sequence=['#4A90E2'])
            st.plotly_chart(fig_rain, use_container_width=True)
        with col_rain_txt:
            st.write("The distribution shows minimal precipitation at the start of the year with a sharp rise during monsoon months.")

    with tab3:
        st.subheader("Monthly Average Temperature (Orange Bar Chart)")
        monthly_avg = df_raw.groupby('Month')['Temperature'].mean().reset_index()
        fig_monthly = px.bar(monthly_avg, x='Month', y='Temperature', color_discrete_sequence=['#FFA500'])
        st.plotly_chart(fig_monthly, use_container_width=True)
        st.write("This helps identify seasonal warming and cooling patterns throughout 2023.")

    with tab4:
        st.subheader("Feature Correlation Heatmap")
        fig_heat, ax_heat = plt.subplots(figsize=(10, 6))
        cols = ['Temperature', 'Rainfall', 'Humidity', 'WindSpeed']
        sns.heatmap(df[cols].corr(), annot=True, cmap='RdBu', center=0)
        st.pyplot(fig_heat)

    # 6. DATA EXPORT
    st.divider()
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button("📥 Download Filtered Results (CSV)", data=csv, file_name=f"{selected_area}_weather.csv")

except Exception as e:
    st.error(f"Dashboard Error: {e}")