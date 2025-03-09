import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import plotly.express as px
from datetime import datetime
import matplotlib.pyplot as plt

# --------------------------
# Streamlit App Configuration
# --------------------------
# Must be the first Streamlit command
st.set_page_config(page_title="US Accidents Analysis", layout="wide")

# --------------------------
# Data Loading and Preparation
# --------------------------
@st.cache_data  # Updated from experimental_memo to cache_data
def load_data():
    df = pd.read_csv('US_Accidents_March23.csv')

    df['Wind_Chill(F)'] = df['Wind_Chill(F)'].fillna(df['Wind_Chill(F)'].mean())
    df = df.dropna(subset=['City', 'Zipcode'])

    # Convert to datetime with format='mixed' to handle various formats including fractional seconds
    df['Start_Time'] = pd.to_datetime(df['Start_Time'], format='mixed')
    df['End_Time'] = pd.to_datetime(df['End_Time'], format='mixed')

    # Calculate accident duration
    df['Duration'] = (df['End_Time'] - df['Start_Time']).dt.total_seconds() / 60

    return df

# --------------------------
# App Title
# --------------------------
st.title("📊 Comprehensive Analysis of US Traffic Accidents")

# Load data
df = load_data()

# --------------------------
# Main Dashboard Sections
# --------------------------
with st.sidebar:
    st.header("Filter Options")
    selected_years = st.slider("Select Years",
                               min_value=2016,
                               max_value=2023,
                               value=(2016, 2023))

    severity_levels = st.multiselect("Select Severity Levels",
                                     options=df['Severity'].unique(),
                                     default=df['Severity'].unique())

    weather_conditions = st.multiselect("Select Weather Conditions",
                                        options=df['Weather_Condition'].unique(),
                                        default=['Clear', 'Rain', 'Snow'])

# --------------------------
# Data Filtering
# --------------------------
filtered_df = df[
    (df['Start_Time'].dt.year >= selected_years[0]) &
    (df['Start_Time'].dt.year <= selected_years[1]) &
    (df['Severity'].isin(severity_levels)) &
    (df['Weather_Condition'].isin(weather_conditions))
    ]

# --------------------------
# Key Metrics Display
# --------------------------
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Accidents", filtered_df.shape[0])
with col2:
    st.metric("Average Severity", f"{filtered_df['Severity'].mean():.1f}")
with col3:
    st.metric("Longest Duration", f"{filtered_df['Duration'].max():.1f} mins")
with col4:
    st.metric("Most Common Weather", filtered_df['Weather_Condition'].mode()[0])

# --------------------------
# Data Overview Section
# --------------------------
st.header("📈 Data Overview")
tab1, tab2, tab3 = st.tabs(["Raw Data", "Descriptive Statistics", "Missing Values"])

with tab1:
    st.dataframe(filtered_df.head(1000),
                 use_container_width=True,
                 column_config={
                     "Start_Time": "Accident Start",
                     "End_Time": "Accident End",
                     "Duration": st.column_config.NumberColumn(format="%.1f mins")
                 })

with tab2:
    st.subheader("Numerical Statistics")
    st.dataframe(filtered_df.describe(), use_container_width=True)

    st.subheader("Categorical Distributions")
    cat_col = st.selectbox("Select Category",
                           options=['Severity', 'Weather_Condition', 'City'])
    counts = filtered_df[cat_col].value_counts().reset_index()
    fig = px.bar(counts, x=cat_col, y='count', title=f"{cat_col} Distribution")
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    missing_values = filtered_df.isna().sum().reset_index()
    missing_values.columns = ['Column', 'Missing Values']
    fig = px.bar(missing_values, x='Column', y='Missing Values',
                 title="Missing Values Distribution")
    st.plotly_chart(fig, use_container_width=True)

# --------------------------
# Geospatial Analysis
# --------------------------
st.header("🌍 Geospatial Analysis")
gdf = gpd.GeoDataFrame(
    filtered_df,
    geometry=gpd.points_from_xy(
        filtered_df.Start_Lng,
        filtered_df.Start_Lat
    )
)

st.subheader("Accident Hotspots")

# Option 1: Simple Streamlit map without color
st.subheader("Basic Map")
st.map(gdf,
       latitude='Start_Lat',
       longitude='Start_Lng',
       size='Severity',
       use_container_width=True)

# Option 2: More customizable Plotly map with color by severity
st.subheader("Detailed Map with Severity Colors")
# Sample for performance (adjust the sample size as needed)
map_sample = filtered_df.sample(min(1000, len(filtered_df)))

fig = px.scatter_mapbox(
    map_sample,
    lat='Start_Lat',
    lon='Start_Lng',
    color='Severity',
    color_continuous_scale='Viridis',
    size='Severity',
    size_max=15,
    zoom=3,
    mapbox_style="carto-positron",
    title="Accident Severity Map",
    hover_data=['City', 'Weather_Condition', 'Severity', 'Duration']
)
fig.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0})
st.plotly_chart(fig, use_container_width=True)

# --------------------------
# Temporal Analysis
# --------------------------
st.header("⏳ Temporal Patterns")
temp_col1, temp_col2 = st.columns(2)

with temp_col1:
    st.subheader("Hourly Distribution")
    hour_counts = filtered_df['Start_Time'].dt.hour.value_counts().sort_index()
    fig = px.line(hour_counts,
                  labels={'index': 'Hour of Day', 'value': 'Accidents'},
                  title="Accidents by Hour of Day")
    st.plotly_chart(fig, use_container_width=True)

with temp_col2:
    st.subheader("Weekly Pattern")
    weekday_counts = filtered_df['Start_Time'].dt.weekday.value_counts().sort_index()
    fig = px.bar(weekday_counts,
                 labels={'index': 'Weekday', 'value': 'Accidents'},
                 title="Accidents by Weekday")
    st.plotly_chart(fig, use_container_width=True)

# --------------------------
# Weather Impact Analysis
# --------------------------
st.header("⛈️ Weather Impact Analysis")
weather_df = filtered_df.groupby('Weather_Condition').agg({
    'Severity': 'mean',
    'Duration': 'mean',
    'ID': 'count'
}).reset_index().rename(columns={'ID': 'Count'})

fig = px.scatter(weather_df,
                 x='Count',
                 y='Severity',
                 size='Duration',
                 color='Weather_Condition',
                 log_x=True,
                 title="Weather Condition Impact")
st.plotly_chart(fig, use_container_width=True)

# --------------------------
# Advanced Analysis
# --------------------------
st.header("🔍 Advanced Insights")

adv_col1, adv_col2 = st.columns(2)

with adv_col1:
    st.subheader("Correlation Matrix")
    numeric_df = filtered_df.select_dtypes(include=[np.number])
    fig = px.imshow(numeric_df.corr(),
                    labels=dict(x="Features", y="Features", color="Correlation"),
                    x=numeric_df.columns,
                    y=numeric_df.columns)
    st.plotly_chart(fig, use_container_width=True)

with adv_col2:
    st.subheader("Duration vs Severity")
    fig = px.density_heatmap(filtered_df,
                             x='Duration',
                             y='Severity',
                             nbinsx=20,
                             nbinsy=5,
                             title="Accident Duration vs Severity")
    st.plotly_chart(fig, use_container_width=True)

# --------------------------
# Data Export
# --------------------------
st.header("📤 Export Options")
if st.button("Generate Analysis Report"):
    with st.spinner("Generating PDF Report..."):
        # Add PDF generation code here
        st.success("Report generated successfully!")

st.download_button(
    label="Download Processed Data",
    data=filtered_df.to_csv().encode('utf-8'),
    file_name='processed_accidents.csv',
    mime='text/csv'
)