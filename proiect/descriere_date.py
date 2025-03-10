#Tema: Metode de tratare a valorilor lipsa (3 metode). Grupari pe date. Clusterizare dupa zona pentru harta
#Surprinderea valorilor extreme si tratarea lor. Sa schimbam graficul (de corelatie) cu seaborn. Buguri de fixat
#Analize de DSAD (Factorial, Clusterizare, ACP, Discriminanta)

#Modularizare : definire de functie (ca e un singur fisier cu tot codul)
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
# App Title and Dataset Description
# --------------------------
st.title("📊 Comprehensive Analysis of US Traffic Accidents")

with st.expander("📋 Dataset Description"):
    st.markdown("""
    ### Data Dictionary

    This dataset contains information about traffic accidents across the United States between 2016 and 2023. Here's what each column represents:

    #### Identification
    - **ID**: Unique identifier of the accident record
    - **Source**: Source of raw accident data

    #### Accident Details
    - **Severity**: Severity of the accident (1-4), where 1 indicates least impact on traffic (short delay) and 4 indicates significant impact (long delay)
    - **Start_Time**: Start time of the accident in local time zone
    - **End_Time**: When the impact of accident on traffic flow was dismissed
    - **Distance(mi)**: Length of the road affected by the accident in miles
    - **Description**: Human-provided description of the accident

    #### Location Information
    - **Start_Lat/Start_Lng**: GPS coordinates of the start point
    - **End_Lat/End_Lng**: GPS coordinates of the end point
    - **Street**: Street name
    - **City**: City name
    - **County**: County name
    - **State**: State abbreviation
    - **Zipcode**: ZIP code
    - **Country**: Country (US)
    - **Timezone**: Timezone based on location (eastern, central, etc.)

    #### Weather Conditions
    - **Airport_Code**: Closest airport-based weather station
    - **Weather_Timestamp**: Time of weather observation
    - **Temperature(F)**: Temperature in Fahrenheit
    - **Wind_Chill(F)**: Wind chill in Fahrenheit
    - **Humidity(%)**: Humidity percentage
    - **Pressure(in)**: Air pressure in inches
    - **Visibility(mi)**: Visibility in miles
    - **Wind_Direction**: Wind direction
    - **Wind_Speed(mph)**: Wind speed in mph
    - **Precipitation(in)**: Precipitation amount in inches
    - **Weather_Condition**: Weather condition (rain, snow, etc.)

    #### Point of Interest (POI) Annotations
    These boolean fields indicate presence of various features near the accident:
    - **Amenity**, **Bump**, **Crossing**, **Give_Way**, **Junction**, **No_Exit**
    - **Railway**, **Roundabout**, **Station**, **Stop**
    - **Traffic_Calming**, **Traffic_Signal**, **Turning_Loop**

    #### Time of Day Indicators
    - **Sunrise_Sunset**: Day or night based on sunrise/sunset
    - **Civil_Twilight**: Day or night based on civil twilight
    - **Nautical_Twilight**: Day or night based on nautical twilight
    - **Astronomical_Twilight**: Day or night based on astronomical twilight
    """)

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
st.markdown("### 📌 Key Metrics")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Total Accidents", f"{filtered_df.shape[0]:,}")
with col2:
    st.metric("Average Severity", f"{filtered_df['Severity'].mean():.1f}/4")
with col3:
    avg_duration = filtered_df['Duration'].mean()
    st.metric("Average Duration", f"{avg_duration:.1f} mins")
with col4:
    most_common_weather = filtered_df['Weather_Condition'].mode()[0]
    # Truncate if too long
    display_weather = most_common_weather if len(most_common_weather) < 15 else most_common_weather[:12] + "..."
    st.metric("Most Common Weather", display_weather)

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
    # Create a more readable version of the describe table
    stats_df = filtered_df.describe().round(2)
    st.dataframe(stats_df, use_container_width=True)

    st.subheader("Categorical Distributions")
    cat_options = {
        'Severity': 'Accident Severity Level',
        'Weather_Condition': 'Weather Condition',
        'City': 'City',
        'State': 'State',
        'Sunrise_Sunset': 'Time of Day (Sunrise/Sunset)'
    }

    cat_col = st.selectbox("Select Category to Analyze",
                           options=list(cat_options.keys()),
                           format_func=lambda x: cat_options[x])

    # For better visualization, limit to top categories for fields with many values
    if cat_col in ['City', 'Weather_Condition']:
        counts = filtered_df[cat_col].value_counts().head(20).reset_index()
        title = f"Top 20 {cat_options[cat_col]}s by Accident Count"
    else:
        counts = filtered_df[cat_col].value_counts().reset_index()
        title = f"{cat_options[cat_col]} Distribution"

    fig = px.bar(counts, x=cat_col, y='count', title=title)

    # Rotate x-axis labels for better readability if needed
    if cat_col in ['City', 'Weather_Condition']:
        fig.update_layout(xaxis_tickangle=-45)

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

    # Create hour labels with AM/PM format
    hour_labels = {hr: f"{hr}:00" + (" AM" if hr < 12 else " PM") for hr in range(24)}
    hour_labels[0] = "12:00 AM"
    hour_labels[12] = "12:00 PM"

    # Convert to DataFrame for better labeling
    hour_df = pd.DataFrame({'hour': hour_counts.index, 'count': hour_counts.values})
    hour_df['hour_label'] = hour_df['hour'].map(hour_labels)

    fig = px.line(hour_df,
                  x='hour_label',
                  y='count',
                  labels={'hour_label': 'Hour of Day', 'count': 'Number of Accidents'},
                  title="Accidents by Hour of Day",
                  category_orders={"hour_label": [hour_labels[hr] for hr in range(24)]})

    # Add markers to make the line chart more readable
    fig.update_traces(mode='lines+markers')
    st.plotly_chart(fig, use_container_width=True)

with temp_col2:
    st.subheader("Weekly Pattern")
    # Map weekday numbers to names
    weekday_names = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday',
                     3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'}

    # Create series with weekday numbers
    weekday_counts = filtered_df['Start_Time'].dt.weekday.value_counts().sort_index()

    # Convert to DataFrame and map numbers to names
    weekday_df = pd.DataFrame({'weekday_num': weekday_counts.index,
                               'count': weekday_counts.values})
    weekday_df['weekday'] = weekday_df['weekday_num'].map(weekday_names)

    # Create bar chart with weekday names
    fig = px.bar(weekday_df,
                 x='weekday',
                 y='count',
                 labels={'weekday': 'Day of Week', 'count': 'Number of Accidents'},
                 title="Accidents by Weekday",
                 category_orders={"weekday": list(weekday_names.values())})
    st.plotly_chart(fig, use_container_width=True)

# --------------------------
# Weather Impact Analysis
# --------------------------
st.header("⛈️ Weather Impact Analysis")

# Limit to top weather conditions for better visualization
top_weather_conditions = filtered_df['Weather_Condition'].value_counts().head(15).index.tolist()
weather_filtered = filtered_df[filtered_df['Weather_Condition'].isin(top_weather_conditions)]

weather_df = weather_filtered.groupby('Weather_Condition').agg({
    'Severity': 'mean',
    'Duration': 'mean',
    'ID': 'count'
}).reset_index().rename(columns={'ID': 'Accident Count'})

# Sort by count for better visualization
weather_df = weather_df.sort_values('Accident Count', ascending=False)

fig = px.scatter(weather_df,
                 x='Accident Count',
                 y='Severity',
                 size='Duration',
                 color='Weather_Condition',
                 hover_data=['Weather_Condition', 'Accident Count', 'Severity', 'Duration'],
                 labels={
                     'Accident Count': 'Number of Accidents',
                     'Severity': 'Average Severity (1-4)',
                     'Duration': 'Average Duration (minutes)'
                 },
                 title="Impact of Weather Conditions on Accidents")

st.plotly_chart(fig, use_container_width=True)

# Add a bar chart showing distribution of severity by weather condition
st.subheader("Average Severity by Weather Condition")
fig2 = px.bar(weather_df,
              x='Weather_Condition',
              y='Severity',
              color='Severity',
              color_continuous_scale='RdYlGn_r',  # Red for high severity, green for low
              labels={'Weather_Condition': 'Weather Condition', 'Severity': 'Average Severity (1-4)'},
              title="Average Accident Severity by Weather Condition")
fig2.update_layout(xaxis_tickangle=-45)
st.plotly_chart(fig2, use_container_width=True)

# --------------------------
# POI Feature Analysis
# --------------------------
st.header("🚦 Points of Interest Analysis")
st.markdown("Analyzing accident correlations with nearby infrastructure features")

# Get all the POI columns
poi_columns = ['Amenity', 'Bump', 'Crossing', 'Give_Way', 'Junction',
               'No_Exit', 'Railway', 'Roundabout', 'Station', 'Stop',
               'Traffic_Calming', 'Traffic_Signal', 'Turning_Loop']

# Calculate percentage of accidents near each POI
poi_percentages = {}
for col in poi_columns:
    true_count = filtered_df[col].sum()
    total_count = len(filtered_df)
    percentage = (true_count / total_count) * 100
    poi_percentages[col] = percentage

# Create DataFrame for visualization
poi_df = pd.DataFrame({
    'POI_Feature': list(poi_percentages.keys()),
    'Percentage': list(poi_percentages.values())
}).sort_values('Percentage', ascending=False)

# Visualize as horizontal bar chart
fig = px.bar(poi_df,
             y='POI_Feature',
             x='Percentage',
             orientation='h',
             labels={'POI_Feature': 'Infrastructure Feature', 'Percentage': 'Percentage of Accidents (%)'},
             title='Percentage of Accidents Near Different Infrastructure Features',
             color='Percentage',
             color_continuous_scale='Blues')

st.plotly_chart(fig, use_container_width=True)

# Add severity analysis by POI
st.subheader("Average Accident Severity by Infrastructure Feature")

poi_severity = {}
for col in poi_columns:
    # Average severity when feature is present
    severity_true = filtered_df[filtered_df[col] == True]['Severity'].mean()
    # Average severity when feature is not present
    severity_false = filtered_df[filtered_df[col] == False]['Severity'].mean()

    poi_severity[col] = {
        'With Feature': severity_true,
        'Without Feature': severity_false,
        'Difference': severity_true - severity_false
    }

# Create DataFrame for visualization
severity_df = pd.DataFrame(poi_severity).T.reset_index()
severity_df = pd.melt(severity_df,
                      id_vars=['index'],
                      value_vars=['With Feature', 'Without Feature'],
                      var_name='Presence',
                      value_name='Average Severity')
severity_df.rename(columns={'index': 'POI Feature'}, inplace=True)

# Visualization
fig2 = px.bar(severity_df,
              x='POI Feature',
              y='Average Severity',
              color='Presence',
              barmode='group',
              labels={'POI Feature': 'Infrastructure Feature', 'Average Severity': 'Average Severity (1-4)'},
              title='Comparison of Accident Severity With and Without Infrastructure Features')

fig2.update_layout(xaxis_tickangle=-45)
st.plotly_chart(fig2, use_container_width=True)

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