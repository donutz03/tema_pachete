import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.impute import KNNImputer
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler, LabelEncoder, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import statsmodels.api as sm
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from scalare_utils import aplica_scalare, adauga_sectiune_scalare
import plotly.graph_objects as go
from shapely.geometry import Point
import contextily as ctx
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium

st.set_page_config(page_title="US Accidents Analysis", layout="wide")

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

@st.cache_data
def load_data():
    df = pd.read_csv('US_Accidents_Sample_1000_Per_Year.csv')

    # Convertim coloanele de timp
    df['Start_Time'] = pd.to_datetime(df['Start_Time'], errors='coerce')
    df['End_Time'] = pd.to_datetime(df['End_Time'], errors='coerce')

    # Calculăm durata
    df['Duration'] = (df['End_Time'] - df['Start_Time']).dt.total_seconds() / 60

    # Convertim coloanele de tip object în string
    for col in df.select_dtypes(include='object').columns:
        if not pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].astype(str)

    return df

st.title("📊 Analiza Exploratorie a Accidentelor Rutiere")

with st.sidebar:
    st.header("Meniu Principal")
    menu = st.radio(
        "Selectează secțiunea:",
        ["Analiză Generală", "Tratarea Valorilor Lipsă", "Identificarea Valorilor Extreme", 
         "Grupări și Corelații", "Scalarea Datelor", "Codificare și Regresie", 
         "Clusterizare", "Analiză Geografică"]
    )

    st.header("Filtrare Date")
    selected_years = st.slider("Selectează Anii",
                               min_value=2016,
                               max_value=2023,
                               value=(2023, 2023))

    st.markdown("---")
    st.markdown("### Alte filtre")
    severity_levels = st.multiselect("Niveluri de Severitate",
                                     options=[1, 2, 3, 4],
                                     default=[1, 2, 3, 4])

df = load_data()

filtered_df = df[
    (df['Start_Time'].dt.year >= selected_years[0]) &
    (df['Start_Time'].dt.year <= selected_years[1]) &
    (df['Severity'].isin(severity_levels))
    ]

if menu == "Analiză Generală":
    st.header("📊 Informații despre setul de date")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Număr de accidente", f"{filtered_df.shape[0]:,}")
    with col2:
        st.metric("Număr de coloane", f"{filtered_df.shape[1]}")
    with col3:
        st.metric("Perioada acoperită",
                  f"{filtered_df['Start_Time'].dt.year.min()} - {filtered_df['Start_Time'].dt.year.max()}")

    st.subheader("Primele înregistrări")
    st.dataframe(filtered_df.head(10), use_container_width=True)

    st.subheader("Informații despre tipurile de date")
    data_types = pd.DataFrame({
        'Coloană': filtered_df.dtypes.index,
        'Tip': filtered_df.dtypes.values,
        'Valori Nule': filtered_df.isna().sum().values,
        'Procent Nule': (filtered_df.isna().sum().values / len(filtered_df) * 100).round(2)
    })
    data_types['Tip'] = data_types['Tip'].astype(str)
    st.dataframe(data_types, hide_index=True)

    st.subheader("Statistici de bază")
    stats_df = filtered_df.describe().round(2).astype(str)
    st.dataframe(stats_df, hide_index=True)

elif menu == "Tratarea Valorilor Lipsă":
    st.header("🧩 Tratarea Valorilor Lipsă")

    na_cols = filtered_df.columns[filtered_df.isna().any()].tolist()

    if not na_cols:
        st.info("Nu există valori lipsă în datele filtrate!")
    else:
        selected_col = st.selectbox("Selectează coloana pentru tratarea valorilor lipsă", na_cols)

        st.subheader(f"Vizualizarea valorilor lipsă pentru coloana {selected_col}")

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total valori lipsă", filtered_df[selected_col].isna().sum())
            st.metric("Procent valori lipsă",
                      f"{(filtered_df[selected_col].isna().sum() / len(filtered_df) * 100):.2f}%")

        with col2:
            if pd.api.types.is_numeric_dtype(filtered_df[selected_col]):
                fig, ax = plt.subplots(figsize=(6, 3))
                filtered_df[selected_col].hist(ax=ax)
                st.pyplot(fig)

        st.subheader("Metode de tratare a valorilor lipsă")

        tabs = st.tabs(
            ["Metoda 1: Înlocuire cu media/mediana/mod", "Metoda 2: Înlocuire cu KNN", "Metoda 3: Interpolare"])

        with tabs[0]:
            st.markdown("#### Înlocuire cu statistici")

            if pd.api.types.is_numeric_dtype(filtered_df[selected_col]):
                method = st.radio("Alege metoda de înlocuire:", ["Media", "Mediana"])

                if method == "Media":
                    replace_value = filtered_df[selected_col].mean()
                    df_replaced = filtered_df.copy()
                    df_replaced[selected_col] = df_replaced[selected_col].fillna(replace_value)

                    st.success(f"Valorile lipsă au fost înlocuite cu media: {replace_value:.2f}")

                elif method == "Mediana":
                    replace_value = filtered_df[selected_col].median()
                    df_replaced = filtered_df.copy()
                    df_replaced[selected_col] = df_replaced[selected_col].fillna(replace_value)

                    st.success(f"Valorile lipsă au fost înlocuite cu mediana: {replace_value:.2f}")

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("##### Înainte de înlocuire")
                    st.dataframe(filtered_df[[selected_col]].describe(), use_container_width=True)

                with col2:
                    st.markdown("##### După înlocuire")
                    st.dataframe(df_replaced[[selected_col]].describe(), use_container_width=True)

            else:
                mode_series = filtered_df[selected_col].mode()
                if not mode_series.empty:
                    mode_value = mode_series[0]
                    df_replaced = filtered_df.copy()
                    df_replaced[selected_col] = df_replaced[selected_col].fillna(mode_value)
                    st.success(f"Pentru coloana categorică, valorile lipsă au fost înlocuite cu modul: {mode_value}")
                else:
                    st.warning("Nu există o valoare mod pentru această coloană. Nu s-a efectuat nicio înlocuire.")
                    df_replaced = filtered_df.copy()

        with tabs[1]:
            st.markdown("#### Înlocuire cu KNN")

            if pd.api.types.is_numeric_dtype(filtered_df[selected_col]):
                k_neighbors = st.slider("Număr de vecini (K)", 1, 10, 5)

                numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns.tolist()

                # Selectăm doar câteva coloane numerice pentru demonstrație
                cols_for_imputation = st.multiselect(
                    "Selectează coloanele pentru imputare (se recomandă coloane corelate)",
                    numeric_cols,
                    default=[selected_col] + [c for c in numeric_cols if c != selected_col][:2]
                )

                if len(cols_for_imputation) > 1:
                    imputer = KNNImputer(n_neighbors=k_neighbors)

                    # Extragem doar coloanele relevante și eliminăm rândurile unde toate valorile sunt NA
                    subset_df = filtered_df[cols_for_imputation].copy()
                    subset_df = subset_df.dropna(how='all')

                    # Aplicăm imputarea
                    imputed_array = imputer.fit_transform(subset_df)
                    imputed_df = pd.DataFrame(imputed_array, columns=cols_for_imputation)

                    st.success(f"Valorile lipsă au fost înlocuite folosind metoda KNN cu {k_neighbors} vecini")

                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("##### Înainte de înlocuire")
                        st.dataframe(filtered_df[selected_col].describe(), use_container_width=True)

                    with col2:
                        st.markdown("##### După înlocuire")
                        st.dataframe(imputed_df[selected_col].describe(), use_container_width=True)
                else:
                    st.warning("Selectați cel puțin 2 coloane pentru imputare KNN")
            else:
                st.warning("Metoda KNN este aplicabilă doar pentru coloane numerice")

        with tabs[2]:
            st.markdown("#### Interpolare")

            if pd.api.types.is_numeric_dtype(filtered_df[selected_col]):
                method = st.radio("Metoda de interpolare:", ["Linear", "Polynomial", "Spline"])

                df_interpolated = filtered_df.copy()

                if method == "Linear":
                    df_interpolated[selected_col] = df_interpolated[selected_col].interpolate(method='linear')
                    st.success("Interpolare liniară aplicată")

                elif method == "Polynomial":
                    df_interpolated[selected_col] = df_interpolated[selected_col].interpolate(method='polynomial',
                                                                                              order=2)
                    st.success("Interpolare polinomială de gradul 2 aplicată")

                elif method == "Spline":
                    df_interpolated[selected_col] = df_interpolated[selected_col].interpolate(method='spline', order=3)
                    st.success("Interpolare spline cubică aplicată")

                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("##### Înainte de interpolare")
                    st.dataframe(filtered_df[[selected_col]].head(20), use_container_width=True)

                with col2:
                    st.markdown("##### După interpolare")
                    st.dataframe(df_interpolated[[selected_col]].head(20), use_container_width=True)

                st.markdown("##### Comparație statistici")
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("Înainte")
                    st.dataframe(filtered_df[selected_col].describe(), use_container_width=True)

                with col2:
                    st.markdown("După")
                    st.dataframe(df_interpolated[selected_col].describe(), use_container_width=True)
            else:
                st.warning("Interpolarea este aplicabilă doar pentru coloane numerice")

elif menu == "Identificarea Valorilor Extreme":
    st.header("🔍 Identificarea și Tratarea Valorilor Extreme")

    numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns.tolist()

    col_for_outlier = st.selectbox("Selectează coloana pentru analiza outlierilor", numeric_cols)

    st.subheader(f"Analiză outlieri pentru {col_for_outlier}")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### Statistici")
        # Calculează outlieri folosind IQR
        Q1 = filtered_df[col_for_outlier].quantile(0.25)
        Q3 = filtered_df[col_for_outlier].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers = \
        filtered_df[(filtered_df[col_for_outlier] < lower_bound) | (filtered_df[col_for_outlier] > upper_bound)][
            col_for_outlier]

        st.metric("Număr de outlieri", outliers.count())
        st.metric("Procent outlieri", f"{(outliers.count() / filtered_df[col_for_outlier].count() * 100):.2f}%")
        st.metric("Limita inferioară", f"{lower_bound:.2f}")
        st.metric("Limita superioară", f"{upper_bound:.2f}")

        if outliers.count() > 0:
            st.metric("Minim outlieri", f"{outliers.min():.2f}")
            st.metric("Maxim outlieri", f"{outliers.max():.2f}")

    with col2:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(x=filtered_df[col_for_outlier], ax=ax)
        ax.set_title(f"Boxplot pentru {col_for_outlier}")
        st.pyplot(fig)

        # Histogramă cu distribuție
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(filtered_df[col_for_outlier], kde=True, ax=ax)
        ax.axvline(lower_bound, color='r', linestyle='--')
        ax.axvline(upper_bound, color='r', linestyle='--')
        ax.set_title(f"Histogramă cu limite outlieri pentru {col_for_outlier}")
        st.pyplot(fig)

    st.subheader("Metode de tratare a outlierilor")

    outlier_method = st.radio(
        "Selectează metoda de tratare:",
        ["Vizualizare fără tratare", "Înlocuire cu limite", "Transformare logaritmică", "Înlocuire cu valori calculate"]
    )

    if outlier_method == "Vizualizare fără tratare":
        st.dataframe(filtered_df[[col_for_outlier]].describe(), use_container_width=True)

        # Prezintă top 10 valori extreme
        if outliers.count() > 0:
            st.subheader("Top 10 valori extreme")
            extreme_values = outliers.sort_values(ascending=False).head(10)
            st.dataframe(pd.DataFrame(extreme_values), use_container_width=True)

    elif outlier_method == "Înlocuire cu limite":
        df_capped = filtered_df.copy()

        df_capped[col_for_outlier] = df_capped[col_for_outlier].clip(lower=lower_bound, upper=upper_bound)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### Înainte de tratare")
            st.dataframe(filtered_df[col_for_outlier].describe(), use_container_width=True)

            fig, ax = plt.subplots(figsize=(8, 4))
            sns.boxplot(x=filtered_df[col_for_outlier], ax=ax)
            st.pyplot(fig)

        with col2:
            st.markdown("##### După înlocuire cu limite")
            st.dataframe(df_capped[col_for_outlier].describe(), use_container_width=True)

            fig, ax = plt.subplots(figsize=(8, 4))
            sns.boxplot(x=df_capped[col_for_outlier], ax=ax)
            st.pyplot(fig)

    elif outlier_method == "Transformare logaritmică":
        if (filtered_df[col_for_outlier] <= 0).any():
            st.warning(
                "Transformarea logaritmică necesită valori pozitive. Adăugăm o constantă pentru a face toate valorile pozitive.")
            min_val = filtered_df[col_for_outlier].min()
            constant = abs(min_val) + 1 if min_val <= 0 else 0

            df_log = filtered_df.copy()
            df_log[col_for_outlier] = np.log(df_log[col_for_outlier] + constant)

            st.success(
                f"Am adăugat constanta {constant} pentru a face toate valorile pozitive înainte de transformarea log")
        else:
            df_log = filtered_df.copy()
            df_log[col_for_outlier] = np.log(df_log[col_for_outlier])

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### Distribuția originală")
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.histplot(filtered_df[col_for_outlier], kde=True, ax=ax)
            st.pyplot(fig)

        with col2:
            st.markdown("##### Distribuția după transformarea log")
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.histplot(df_log[col_for_outlier], kde=True, ax=ax)
            st.pyplot(fig)

        st.dataframe(df_log[col_for_outlier].describe(), use_container_width=True)

    elif outlier_method == "Înlocuire cu valori calculate":
        replace_method = st.radio(
            "Metodă de înlocuire:",
            ["Medie", "Mediană", "Calcul bazat pe percentile"]
        )

        df_replaced = filtered_df.copy()

        if replace_method == "Medie":
            # Excludem outlieri din calculul mediei
            mean_no_outliers = filtered_df[(filtered_df[col_for_outlier] >= lower_bound) &
                                           (filtered_df[col_for_outlier] <= upper_bound)][col_for_outlier].mean()

            # Înlocuim doar outlieri
            df_replaced.loc[(df_replaced[col_for_outlier] < lower_bound), col_for_outlier] = mean_no_outliers
            df_replaced.loc[(df_replaced[col_for_outlier] > upper_bound), col_for_outlier] = mean_no_outliers

            st.success(f"Outlieri înlocuiți cu media fără outlieri: {mean_no_outliers:.2f}")

        elif replace_method == "Mediană":
            median = filtered_df[col_for_outlier].median()

            df_replaced.loc[(df_replaced[col_for_outlier] < lower_bound), col_for_outlier] = median
            df_replaced.loc[(df_replaced[col_for_outlier] > upper_bound), col_for_outlier] = median

            st.success(f"Outlieri înlocuiți cu mediana: {median:.2f}")

        elif replace_method == "Calcul bazat pe percentile":
            p10 = filtered_df[col_for_outlier].quantile(0.10)
            p90 = filtered_df[col_for_outlier].quantile(0.90)

            df_replaced.loc[(df_replaced[col_for_outlier] < lower_bound), col_for_outlier] = p10
            df_replaced.loc[(df_replaced[col_for_outlier] > upper_bound), col_for_outlier] = p90

            st.success(f"Outlieri mici înlocuiți cu percentila 10: {p10:.2f}")
            st.success(f"Outlieri mari înlocuiți cu percentila 90: {p90:.2f}")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### Boxplot înainte de înlocuire")
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.boxplot(x=filtered_df[col_for_outlier], ax=ax)
            st.pyplot(fig)

        with col2:
            st.markdown("##### Boxplot după înlocuire")
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.boxplot(x=df_replaced[col_for_outlier], ax=ax)
            st.pyplot(fig)

        st.subheader("Comparație statistici")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### Înainte")
            st.dataframe(filtered_df[col_for_outlier].describe(), use_container_width=True)

        with col2:
            st.markdown("##### După")
            st.dataframe(df_replaced[col_for_outlier].describe(), use_container_width=True)

elif menu == "Grupări și Corelații":
    st.header("📊 Grupări și Corelații")

    tabs = st.tabs(["Corelații", "Grupări", "Funcții Agregate"])

    with tabs[0]:
        st.subheader("Matricea de corelație")

        numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns.tolist()
        selected_cols = st.multiselect(
            "Selectează coloanele pentru analiza corelațiilor",
            numeric_cols,
            default=numeric_cols[:8]  # Primele 8 coloane numerice
        )

        if not selected_cols:
            st.warning("Selectați cel puțin o coloană!")
        else:
            corr_method = st.radio("Metoda de corelație:", ["Pearson", "Spearman", "Kendall"])

            fig, ax = plt.subplots(figsize=(12, 10))
            corr_matrix = filtered_df[selected_cols].corr(method=corr_method.lower())

            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(corr_matrix, mask=mask, annot=True, fmt=".2f",
                        cmap="coolwarm", ax=ax, cbar_kws={"shrink": .8})

            plt.title(f"Matrice de corelație folosind metoda {corr_method}")
            plt.tight_layout()
            st.pyplot(fig)

            st.subheader("Selectează perechi de coloane pentru analiză")

            col1, col2 = st.columns(2)
            with col1:
                x_col = st.selectbox("Coloana X", selected_cols, index=0)
            with col2:
                y_col = st.selectbox("Coloana Y", [c for c in selected_cols if c != x_col], index=0)

            fig, ax = plt.subplots(figsize=(10, 6))
            sns.regplot(x=filtered_df[x_col], y=filtered_df[y_col], ax=ax)
            plt.title(f"Relația dintre {x_col} și {y_col}")
            plt.tight_layout()
            st.pyplot(fig)

            st.metric(f"Coeficient de corelație {corr_method}",
                      f"{filtered_df[x_col].corr(filtered_df[y_col], method=corr_method.lower()):.3f}")

    with tabs[1]:
        st.subheader("Gruparea datelor")

        # Coloane pentru grupare
        all_cols = filtered_df.columns.tolist()
        categorical_cols = filtered_df.select_dtypes(include=['object', 'category']).columns.tolist()
        date_cols = ['Start_Time']

        # Adăugăm coloane derivate pentru grupare temporală
        filtered_df['Month'] = filtered_df['Start_Time'].dt.month
        filtered_df['DayOfWeek'] = filtered_df['Start_Time'].dt.dayofweek
        filtered_df['Hour'] = filtered_df['Start_Time'].dt.hour

        group_by_options = categorical_cols + ['Month', 'DayOfWeek', 'Hour', 'Severity']

        col1, col2 = st.columns(2)
        with col1:
            groupby_col = st.selectbox("Grupează după", group_by_options)

        with col2:
            agg_col = st.selectbox("Aplică funcție pe coloana",
                                   [c for c in filtered_df.select_dtypes(include=[np.number]).columns if
                                    c != groupby_col])

        agg_func = st.radio("Funcție de agregare", ["Număr", "Medie", "Sumă", "Minim", "Maxim", "Mediană"])

        if agg_func == "Număr":
            result_df = filtered_df.groupby(groupby_col).size().reset_index(name='Număr')
        elif agg_func == "Medie":
            result_df = filtered_df.groupby(groupby_col)[agg_col].mean().reset_index(name=f'Medie {agg_col}')
        elif agg_func == "Sumă":
            result_df = filtered_df.groupby(groupby_col)[agg_col].sum().reset_index(name=f'Sumă {agg_col}')
        elif agg_func == "Minim":
            result_df = filtered_df.groupby(groupby_col)[agg_col].min().reset_index(name=f'Minim {agg_col}')
        elif agg_func == "Maxim":
            result_df = filtered_df.groupby(groupby_col)[agg_col].max().reset_index(name=f'Maxim {agg_col}')
        elif agg_func == "Mediană":
            result_df = filtered_df.groupby(groupby_col)[agg_col].median().reset_index(name=f'Mediană {agg_col}')

        # Sortăm rezultatul
        if agg_func == "Număr":
            result_df = result_df.sort_values(by='Număr', ascending=False)
        else:
            result_df = result_df.sort_values(by=result_df.columns[1], ascending=False)

        st.subheader("Rezultat grupare")
        st.dataframe(result_df, use_container_width=True)

        # Vizualizăm rezultatul cu un grafic
        fig = px.bar(
            result_df.head(20),
            x=groupby_col,
            y=result_df.columns[1],
            title=f"{agg_func} de {agg_col if agg_func != 'Număr' else 'accidente'} grupat după {groupby_col}"
        )
        fig.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig, use_container_width=True)

    with tabs[2]:
        st.subheader("Funcții de agregare multiple")

        all_cols = filtered_df.columns.tolist()
        categorical_cols = filtered_df.select_dtypes(include=['object', 'category']).columns.tolist()
        numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns.tolist()

        col1, col2 = st.columns(2)
        with col1:
            groupby_cols = st.multiselect("Grupează după (selectează una sau mai multe coloane)",
                                          categorical_cols + ['Month', 'DayOfWeek', 'Hour', 'Severity'],
                                          default=[categorical_cols[0] if categorical_cols else 'Severity'])

        with col2:
            agg_cols = st.multiselect("Coloane pentru agregare",
                                      [c for c in numeric_cols if c not in groupby_cols],
                                      default=[numeric_cols[0] if numeric_cols else 'Duration'])

        agg_funcs = st.multiselect("Funcții de agregare",
                                   ["count", "mean", "sum", "min", "max", "median", "std", "var"],
                                   default=["count", "mean"])

        if not groupby_cols:
            st.warning("Selectați cel puțin o coloană pentru grupare!")
        elif not agg_cols:
            st.warning("Selectați cel puțin o coloană pentru agregare!")
        elif not agg_funcs:
            st.warning("Selectați cel puțin o funcție de agregare!")
        else:
            # Construim dicționarul pentru agregare
            agg_dict = {col: agg_funcs for col in agg_cols}

            result_df = filtered_df.groupby(groupby_cols).agg(agg_dict)

            # Resetăm index-ul pentru afișare mai ușoară
            result_df = result_df.reset_index()

            # Afișăm rezultatul
            st.dataframe(result_df, use_container_width=True)

            # Opțiune pentru descărcare
            csv = result_df.to_csv(index=False)
            st.download_button(
                label="Descarcă rezultatele în CSV",
                data=csv,
                file_name=f"grupare_{'_'.join(groupby_cols)}.csv",
                mime="text/csv"
            )

            # Vizualizare grafică pentru prima funcție de agregare și prima coloană
            if len(groupby_cols) == 1 and len(result_df) <= 25:
                agg_col_name = f"{agg_cols[0]}_{agg_funcs[0]}"

                fig = px.bar(
                    result_df.head(25),
                    x=groupby_cols[0],
                    y=agg_col_name,
                    title=f"{agg_funcs[0]} de {agg_cols[0]} grupat după {groupby_cols[0]}"
                )
                fig.update_layout(xaxis_tickangle=-45)
                st.plotly_chart(fig, use_container_width=True)

elif menu == "Scalarea Datelor":
    # Aplicăm funcția pentru vizualizarea și aplicarea metodelor de scalare
    st.header("🔄 Metode de Scalare a Datelor")
    adauga_sectiune_scalare(filtered_df, sidebar=False)

# Adăugăm o nouă opțiune pentru BoxPlot interactiv
st.sidebar.markdown("---")
if st.sidebar.checkbox("Activează BoxPlot Interactiv"):
    st.header("📊 BoxPlot Interactiv")

    numeric_columns = filtered_df.select_dtypes(include=[np.number]).columns.tolist()
    selected_column = st.selectbox("Selectează coloana pentru BoxPlot", numeric_columns)

    col1, col2 = st.columns([2, 1])

    with col1:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(y=filtered_df[selected_column], ax=ax)
        ax.set_title(f"BoxPlot pentru {selected_column}")
        st.pyplot(fig)

    with col2:
        st.markdown("### Statistici")
        stats = filtered_df[selected_column].describe()

        # Calculăm manual IQR pentru limite de outlieri
        Q1 = stats["25%"]
        Q3 = stats["75%"]
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outliers_count = filtered_df[(filtered_df[selected_column] < lower_bound) |
                                     (filtered_df[selected_column] > upper_bound)].shape[0]

        st.metric("Minim", f"{stats['min']:.2f}")
        st.metric("Q1 (25%)", f"{Q1:.2f}")
        st.metric("Mediană", f"{stats['50%']:.2f}")
        st.metric("Q3 (75%)", f"{Q3:.2f}")
        st.metric("Maxim", f"{stats['max']:.2f}")
        st.metric("IQR", f"{IQR:.2f}")
        st.metric("Limita inferioară", f"{lower_bound:.2f}")
        st.metric("Limita superioară", f"{upper_bound:.2f}")
        st.metric("Număr outlieri", f"{outliers_count} ({outliers_count / len(filtered_df) * 100:.1f}%)")

    if st.checkbox("Arată histograma"):
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(filtered_df[selected_column], kde=True, ax=ax)
        plt.axvline(lower_bound, color='r', linestyle='--', label='Limite outlieri')
        plt.axvline(upper_bound, color='r', linestyle='--')
        plt.legend()
        st.pyplot(fig)

elif menu == "Codificare și Regresie":
    st.header("🔢 Codificare și Analiză de Regresie")
    
    tabs = st.tabs(["Codificare Date", "Regresie Logistică", "Regresie Multiplă"])
    
    with tabs[0]:
        st.subheader("Codificare Date")
        
        # Selectăm coloanele categorice pentru codificare
        categorical_cols = filtered_df.select_dtypes(include=['object']).columns.tolist()
        selected_col = st.selectbox("Selectează coloana pentru codificare", categorical_cols)
        
        encoding_method = st.radio("Alege metoda de codificare:", ["Label Encoding", "One-Hot Encoding"])
        
        if encoding_method == "Label Encoding":
            le = LabelEncoder()
            encoded_values = le.fit_transform(filtered_df[selected_col].fillna('Missing'))
            
            # Creăm un DataFrame pentru vizualizare
            encoding_df = pd.DataFrame({
                'Valoare Originală': filtered_df[selected_col].fillna('Missing'),
                'Valoare Codificată': encoded_values
            }).drop_duplicates().sort_values('Valoare Codificată')
            
            st.dataframe(encoding_df, use_container_width=True)
            
        else:  # One-Hot Encoding
            ohe = OneHotEncoder(sparse=False)
            encoded_values = ohe.fit_transform(filtered_df[[selected_col]].fillna('Missing'))
            
            # Creăm un DataFrame pentru vizualizare
            feature_names = [f"{selected_col}_{val}" for val in ohe.categories_[0]]
            encoding_df = pd.DataFrame(encoded_values, columns=feature_names)
            
            st.dataframe(encoding_df.head(), use_container_width=True)
            
            # Vizualizăm distribuția valorilor codificate
            fig = px.bar(encoding_df.sum(), title=f"Distribuția valorilor codificate pentru {selected_col}")
            st.plotly_chart(fig)
    
    with tabs[1]:
        st.subheader("Regresie Logistică")
        
        # Selectăm variabilele pentru regresie logistică
        target_col = st.selectbox("Selectează variabila țintă (binară)", 
                                ['Severity', 'Amenity', 'Bump', 'Crossing', 'Give_Way', 'Junction', 'No_Exit'],
                                key="logistic_target")
        
        feature_cols = st.multiselect("Selectează variabilele predictoare (numerice)",
                                    filtered_df.select_dtypes(include=[np.number]).columns.tolist(),
                                    default=['Temperature(F)', 'Humidity(%)', 'Pressure(in)', 'Visibility(mi)', 'Wind_Speed(mph)'],
                                    key="logistic_features")
        
        if len(feature_cols) > 0:
            # Pregătim datele
            X = filtered_df[feature_cols].fillna(filtered_df[feature_cols].mean())
            y = (filtered_df[target_col] > filtered_df[target_col].median()).astype(int)
            
            # Împărțim datele în set de antrenare și test
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
            
            # Antrenăm modelul
            model = LogisticRegression(max_iter=1000)
            model.fit(X_train, y_train)
            
            # Evaluăm modelul
            y_pred = model.predict(X_test)
            
            st.subheader("Rezultate Model")
            st.text(classification_report(y_test, y_pred))
            
            # Vizualizăm matricea de confuzie
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            st.pyplot(fig)
            
            # Afișăm coeficienții modelului
            coef_df = pd.DataFrame({
                'Feature': feature_cols,
                'Coefficient': model.coef_[0]
            })
            st.dataframe(coef_df.sort_values('Coefficient', ascending=False), use_container_width=True)
    
    with tabs[2]:
        st.subheader("Regresie Multiplă")
        
        # Selectăm variabilele pentru regresie multiplă
        target_col = st.selectbox("Selectează variabila țintă (numerică)", 
                                ['Duration', 'Distance(mi)', 'Temperature(F)', 'Humidity(%)', 'Pressure(in)', 'Visibility(mi)', 'Wind_Speed(mph)'],
                                key="multiple_target")
        
        feature_cols = st.multiselect("Selectează variabilele predictoare (numerice)",
                                    filtered_df.select_dtypes(include=[np.number]).columns.tolist(),
                                    default=['Temperature(F)', 'Humidity(%)', 'Pressure(in)', 'Visibility(mi)', 'Wind_Speed(mph)'],
                                    key="multiple_features")
        
        if len(feature_cols) > 0:
            # Pregătim datele
            X = filtered_df[feature_cols].fillna(filtered_df[feature_cols].mean())
            y = filtered_df[target_col].fillna(filtered_df[target_col].mean())
            
            # Adăugăm constanta pentru statsmodels
            X = sm.add_constant(X)
            
            # Antrenăm modelul
            model = sm.OLS(y, X).fit()
            
            # Afișăm rezultatele
            st.subheader("Rezultate Model")
            st.text(model.summary())
            
            # Vizualizăm reziduurile
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(model.fittedvalues, model.resid)
            ax.axhline(y=0, color='r', linestyle='--')
            ax.set_xlabel('Valori Prezise')
            ax.set_ylabel('Reziduuri')
            st.pyplot(fig)
            
            # Vizualizăm coeficienții
            coef_df = pd.DataFrame({
                'Feature': ['const'] + feature_cols,
                'Coefficient': model.params,
                'P-value': model.pvalues
            })
            st.dataframe(coef_df.sort_values('P-value'), use_container_width=True)

elif menu == "Analiză Geografică":
    st.header("🗺️ Analiză Geografică cu GeoPandas")
    
    # Verificăm dacă avem date valide pentru coordonate
    valid_coords = filtered_df.dropna(subset=['Start_Lat', 'Start_Lng'])
    
    if len(valid_coords) == 0:
        st.warning("Nu există date cu coordonate valide în selecția curentă!")
    else:
        st.subheader(f"Analiza geografică pentru {len(valid_coords)} accidente")
        
        # Creăm geometria punctelor
        geometry = [Point(xy) for xy in zip(valid_coords.Start_Lng, valid_coords.Start_Lat)]
        
        # Creăm GeoDataFrame
        gdf = gpd.GeoDataFrame(valid_coords, geometry=geometry, crs="EPSG:4326")
        
        # Tabs pentru diferite vizualizări
        geo_tabs = st.tabs(["Harta Interactivă", "Analiză pe State", "Analiză pe Zone", "Densitate Accidente"])
        
        with geo_tabs[0]:
            st.subheader("Hartă Interactivă a Accidentelor")
            
            # Selectăm tipul de vizualizare
            map_type = st.radio("Tip hartă:", ["Puncte individuale", "Heatmap", "Clustere"])
            
            # Centrul hărții (media coordonatelor)
            center_lat = valid_coords['Start_Lat'].mean()
            center_lng = valid_coords['Start_Lng'].mean()
            
            # Creăm harta Folium
            m = folium.Map(location=[center_lat, center_lng], zoom_start=5)
            
            if map_type == "Puncte individuale":
                # Alegem ce să colorăm
                color_by = st.selectbox("Colorează după:", ["Severity", "Weather_Condition", "Hour"])
                
                # Definim culori pentru severitate
                severity_colors = {1: 'green', 2: 'yellow', 3: 'orange', 4: 'red'}
                
                # Adăugăm puncte pe hartă
                for idx, row in gdf.iterrows():
                    if color_by == "Severity":
                        color = severity_colors.get(row['Severity'], 'gray')
                    elif color_by == "Weather_Condition":
                        # Pentru condiții meteo folosim o paletă diferită
                        weather_colors = {
                            'Clear': 'blue', 'Cloudy': 'gray', 'Rain': 'lightblue',
                            'Snow': 'white', 'Fog': 'darkgray'
                        }
                        color = weather_colors.get(row['Weather_Condition'], 'black')
                    else:  # Hour
                        hour = row['Start_Time'].hour
                        if 6 <= hour < 12:
                            color = 'orange'  # Dimineața
                        elif 12 <= hour < 18:
                            color = 'yellow'  # După-amiaza
                        elif 18 <= hour < 22:
                            color = 'purple'  # Seara
                        else:
                            color = 'darkblue'  # Noaptea
                    
                    folium.CircleMarker(
                        location=[row.geometry.y, row.geometry.x],
                        radius=5,
                        popup=f"Severity: {row['Severity']}<br>Time: {row['Start_Time']}<br>Weather: {row['Weather_Condition']}",
                        color=color,
                        fill=True,
                        fillColor=color,
                        fillOpacity=0.7
                    ).add_to(m)
            
            elif map_type == "Heatmap":
                # Creăm heatmap
                heat_data = [[row.geometry.y, row.geometry.x] for idx, row in gdf.iterrows()]
                HeatMap(heat_data).add_to(m)
            
            else:  # Clustere
                # Adăugăm clustere de puncte
                marker_cluster = folium.plugins.MarkerCluster().add_to(m)
                
                for idx, row in gdf.iterrows():
                    folium.Marker(
                        location=[row.geometry.y, row.geometry.x],
                        popup=f"Severity: {row['Severity']}<br>Time: {row['Start_Time']}"
                    ).add_to(marker_cluster)
            
            # Afișăm harta
            st_folium(m, width=800, height=600)
        
        with geo_tabs[1]:
            st.subheader("Analiză pe State")
            
            # Grupăm după state
            state_analysis = valid_coords.groupby('State').agg({
                'ID': 'count',
                'Severity': 'mean',
                'Start_Lat': 'mean',
                'Start_Lng': 'mean'
            }).reset_index()
            
            state_analysis.columns = ['State', 'Număr Accidente', 'Severitate Medie', 'Lat', 'Lng']
            state_analysis = state_analysis.sort_values('Număr Accidente', ascending=False)
            
            # Afișăm top 10 state
            st.subheader("Top 10 State după Număr de Accidente")
            st.dataframe(state_analysis.head(10), use_container_width=True)
            
            # Grafic bar
            fig = px.bar(
                state_analysis.head(20),
                x='State',
                y='Număr Accidente',
                color='Severitate Medie',
                title='Top 20 State după Număr de Accidente'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Hartă cu bule pentru state
            st.subheader("Hartă State - Dimensiune după Număr Accidente")
            
            state_map = folium.Map(location=[center_lat, center_lng], zoom_start=4)
            
            for idx, row in state_analysis.iterrows():
                folium.CircleMarker(
                    location=[row['Lat'], row['Lng']],
                    radius=np.sqrt(row['Număr Accidente']) / 2,
                    popup=f"{row['State']}<br>Accidente: {row['Număr Accidente']}<br>Severitate Medie: {row['Severitate Medie']:.2f}",
                    color='red',
                    fill=True,
                    fillOpacity=0.6
                ).add_to(state_map)
            
            st_folium(state_map, width=800, height=600)
        
        with geo_tabs[2]:
            st.subheader("Analiză pe Zone")
            
            # Creăm zone folosind hexbin sau grid
            zone_type = st.radio("Tip de zonă:", ["Grid rectangular", "Hexagoane"])
            
            if zone_type == "Grid rectangular":
                # Creăm grid
                resolution = st.slider("Rezoluție grid (număr celule pe latură)", 10, 50, 20)
                
                # Calculăm limitele
                minx, miny, maxx, maxy = gdf.total_bounds
                
                # Creăm grid-ul
                x_step = (maxx - minx) / resolution
                y_step = (maxy - miny) / resolution
                
                # Creăm celulele grid-ului
                grid_cells = []
                grid_counts = []
                
                for i in range(resolution):
                    for j in range(resolution):
                        cell_minx = minx + i * x_step
                        cell_miny = miny + j * y_step
                        cell_maxx = cell_minx + x_step
                        cell_maxy = cell_miny + y_step
                        
                        # Creăm poligonul celulei
                        cell = gpd.GeoSeries([
                            Point(cell_minx, cell_miny),
                            Point(cell_maxx, cell_miny),
                            Point(cell_maxx, cell_maxy),
                            Point(cell_minx, cell_maxy)
                        ]).unary_union.convex_hull
                        
                        # Numărăm punctele în celulă
                        points_in_cell = gdf[gdf.geometry.within(cell)]
                        count = len(points_in_cell)
                        
                        if count > 0:
                            grid_cells.append(cell)
                            grid_counts.append(count)
                
                # Creăm GeoDataFrame pentru grid
                grid_gdf = gpd.GeoDataFrame({'geometry': grid_cells, 'count': grid_counts})
                
                # Vizualizăm
                fig, ax = plt.subplots(figsize=(12, 8))
                grid_gdf.plot(column='count', cmap='YlOrRd', legend=True, ax=ax)
                ax.set_title('Densitate Accidente pe Grid')
                st.pyplot(fig)
                
            else:  # Hexagoane
                st.info("Pentru o vizualizare hexagonală mai avansată, se recomandă folosirea h3-py sau alte biblioteci specializate.")
                
                # Alternativ, folosim scatter plot cu hexbin
                fig, ax = plt.subplots(figsize=(12, 8))
                hb = ax.hexbin(gdf.geometry.x, gdf.geometry.y, gridsize=30, cmap='YlOrRd')
                cb = fig.colorbar(hb, ax=ax)
                cb.set_label('Număr Accidente')
                ax.set_title('Densitate Accidente - Hexbin')
                st.pyplot(fig)
        
        with geo_tabs[3]:
            st.subheader("Analiză Densitate Accidente")
            
            # Kernel Density Estimation
            st.markdown("### Estimare Densitate Kernel (KDE)")
            
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # Plot KDE
            gdf.plot(ax=ax, alpha=0.5, color='red', markersize=1)
            
            # Adăugăm harta de bază
            try:
                ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron, crs=gdf.crs.to_string())
            except:
                st.warning("Nu s-a putut adăuga harta de bază. Continuăm fără ea.")
            
            ax.set_title('Distribuția Geografică a Accidentelor')
            st.pyplot(fig)
            
            # Analiza pe zone metropolitane
            st.markdown("### Analiza pe Zone Metropolitane")
            
            # Grupăm după oraș și calculăm statistici
            city_analysis = valid_coords.groupby('City').agg({
                'ID': 'count',
                'Severity': 'mean',
                'Distance(mi)': 'mean',
                'Start_Lat': 'mean',
                'Start_Lng': 'mean'
            }).reset_index()
            
            city_analysis.columns = ['City', 'Număr Accidente', 'Severitate Medie', 
                                   'Distanță Medie', 'Lat', 'Lng']
            city_analysis = city_analysis.sort_values('Număr Accidente', ascending=False)
            
            # Afișăm top 20 orașe
            st.subheader("Top 20 Orașe după Număr de Accidente")
            st.dataframe(city_analysis.head(20), use_container_width=True)
            
            # Scatter plot pentru orașe
            fig = px.scatter(
                city_analysis.head(50),
                x='Lng',
                y='Lat',
                size='Număr Accidente',
                color='Severitate Medie',
                hover_name='City',
                title='Top 50 Orașe - Dimensiune după Număr Accidente',
                color_continuous_scale='RdYlBu_r'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            # Analiza temporală geografică
            st.markdown("### Analiza Temporală Geografică")
            
            # Selectăm perioada
            time_period = st.selectbox("Analizează după:", 
                                     ["Ora din zi", "Zi a săptămânii", "Lună", "An"])
            
            if time_period == "Ora din zi":
                valid_coords['Hour'] = valid_coords['Start_Time'].dt.hour
                time_group = 'Hour'
            elif time_period == "Zi a săptămânii":
                valid_coords['DayOfWeek'] = valid_coords['Start_Time'].dt.dayofweek
                time_group = 'DayOfWeek'
            elif time_period == "Lună":
                valid_coords['Month'] = valid_coords['Start_Time'].dt.month
                time_group = 'Month'
            else:  # An
                valid_coords['Year'] = valid_coords['Start_Time'].dt.year
                time_group = 'Year'
            
            # Creăm animație pentru perioada selectată
            time_data = valid_coords.groupby(time_group).agg({
                'ID': 'count',
                'Start_Lat': list,
                'Start_Lng': list
            }).reset_index()
            
            # Afișăm evoluția în timp
            fig = px.scatter_mapbox(
                valid_coords,
                lat='Start_Lat',
                lon='Start_Lng',
                color='Severity',
                animation_frame=time_group,
                zoom=3,
                mapbox_style="carto-positron",
                title=f'Evoluția Accidentelor după {time_period}'
            )
            st.plotly_chart(fig, use_container_width=True)

elif menu == "Clusterizare":
    st.header("🔍 Analiza Clusterizării")
    
    # Selectăm variabilele pentru clusterizare
    numeric_cols = filtered_df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = st.multiselect(
        "Selectează variabilele pentru clusterizare",
        numeric_cols,
        default=['Temperature(F)', 'Humidity(%)', 'Pressure(in)', 'Visibility(mi)', 'Wind_Speed(mph)'],
        key="cluster_features"
    )
    
    if len(feature_cols) > 1:
        # Pregătim datele
        X = filtered_df[feature_cols].fillna(filtered_df[feature_cols].mean())
        
        # Scalăm datele
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Determinăm numărul optim de clustere
        max_clusters = min(10, len(filtered_df) - 1)
        
        # Calculăm metricile pentru diferite numere de clustere
        silhouette_scores = []
        inertia_scores = []
        calinski_scores = []
        
        for n_clusters in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            clusters = kmeans.fit_predict(X_scaled)
            
            # Calculăm scorurile
            silhouette_scores.append(silhouette_score(X_scaled, clusters))
            inertia_scores.append(kmeans.inertia_)
            calinski_scores.append(calinski_harabasz_score(X_scaled, clusters))
        
        # Creăm un DataFrame cu rezultatele
        results_df = pd.DataFrame({
            'Număr Clustere': range(2, max_clusters + 1),
            'Scor Siluetă': silhouette_scores,
            'Inerție': inertia_scores,
            'Scor Calinski-Harabasz': calinski_scores
        })
        
        # Afișăm rezultatele
        st.subheader("Determinarea numărului optim de clustere")
        
        # Creăm un tab pentru fiecare metodă
        tabs = st.tabs(["Scor Siluetă", "Metoda Cotului", "Calinski-Harabasz"])
        
        with tabs[0]:
            st.write("Scorul de siluetă măsoară cât de bine sunt separate clusterele. Valori mai mari indică o clusterizare mai bună.")
            fig = go.Figure()
            
            # Adăugăm inerția
            fig.add_trace(go.Scatter(
                x=results_df['Număr Clustere'],
                y=results_df['Scor Siluetă'],
                name='Scor Siluetă',
                line=dict(color='blue')
            ))
            
            # Configurăm layout-ul
            fig.update_layout(
                title='Scor Siluetă vs Număr Clustere',
                xaxis_title='Număr Clustere',
                yaxis_title='Scor Siluetă'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Găsim numărul optim de clustere bazat pe scorul de siluetă
            optimal_silhouette = results_df.loc[results_df['Scor Siluetă'].idxmax()]
            st.success(f"Numărul optim de clustere bazat pe scorul de siluetă: {int(optimal_silhouette['Număr Clustere'])}")
        
        with tabs[1]:
            st.write("Metoda cotului (elbow method) analizează rata de scădere a inerției. Căutăm 'cotul' în grafic.")
            
            # Calculăm rata de scădere a inerției
            results_df['Rata Scădere'] = results_df['Inerție'].pct_change()
            
            # Creăm figura cu două axe y
            fig = go.Figure()
            
            # Adăugăm inerția
            fig.add_trace(go.Scatter(
                x=results_df['Număr Clustere'],
                y=results_df['Inerție'],
                name='Inerție',
                line=dict(color='blue')
            ))
            
            # Adăugăm rata de scădere
            fig.add_trace(go.Scatter(
                x=results_df['Număr Clustere'],
                y=results_df['Rata Scădere'],
                name='Rata Scădere',
                line=dict(color='red'),
                yaxis='y2'
            ))
            
            # Configurăm layout-ul
            fig.update_layout(
                title='Metoda Cotului (Elbow Method)',
                xaxis_title='Număr Clustere',
                yaxis=dict(
                    title=dict(
                        text='Inerție',
                        font=dict(color='blue')
                    ),
                    tickfont=dict(color='blue')
                ),
                yaxis2=dict(
                    title=dict(
                        text='Rata Scădere',
                        font=dict(color='red')
                    ),
                    tickfont=dict(color='red'),
                    overlaying='y',
                    side='right'
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Găsim punctul de cot
            optimal_elbow = results_df.loc[results_df['Rata Scădere'].idxmin()]
            st.success(f"Sugestie pentru numărul optim de clustere bazat pe metoda cotului: {int(optimal_elbow['Număr Clustere'])}")
        
        with tabs[2]:
            st.write("Scorul Calinski-Harabasz măsoară raportul dintre dispersia inter-cluster și intra-cluster. Valori mai mari indică o clusterizare mai bună.")
            fig = go.Figure()
            
            # Adăugăm scorul Calinski-Harabasz
            fig.add_trace(go.Scatter(
                x=results_df['Număr Clustere'],
                y=results_df['Scor Calinski-Harabasz'],
                name='Scor Calinski-Harabasz',
                line=dict(color='green')
            ))
            
            # Configurăm layout-ul
            fig.update_layout(
                title='Scor Calinski-Harabasz vs Număr Clustere',
                xaxis_title='Număr Clustere',
                yaxis_title='Scor Calinski-Harabasz'
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Găsim numărul optim de clustere bazat pe scorul Calinski-Harabasz
            optimal_calinski = results_df.loc[results_df['Scor Calinski-Harabasz'].idxmax()]
            st.success(f"Numărul optim de clustere bazat pe scorul Calinski-Harabasz: {int(optimal_calinski['Număr Clustere'])}")
        
        # Aplicăm K-means cu numărul optim de clustere
        n_clusters = int(optimal_silhouette['Număr Clustere'])
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(X_scaled)
        
        # Afișăm distribuția punctelor în clustere
        st.subheader("Distribuția punctelor în clustere")
        cluster_distribution = pd.Series(clusters).value_counts().sort_index()
        st.dataframe(cluster_distribution.to_frame('Număr Puncte'), hide_index=True)
        
        # Calculăm scorul de siluetă final
        silhouette_avg = silhouette_score(X_scaled, clusters)
        
        # Adăugăm clusterele la DataFrame
        df_clustered = filtered_df.copy()
        df_clustered['Cluster'] = clusters
        
        # Afișăm statistici despre clustere
        st.subheader("Statistici despre clustere")
        cluster_stats = df_clustered.groupby('Cluster')[feature_cols].mean()
        # Convertim la string pentru a evita probleme de serializare
        cluster_stats_str = cluster_stats.round(2).astype(str)
        st.dataframe(cluster_stats_str, hide_index=True)
        
        # Afișăm scorul de siluetă
        st.metric("Scor de siluetă", f"{silhouette_avg:.3f}")
        
        # Vizualizăm clusterele în spațiul 2D
        st.subheader("Vizualizare clustere")
        
        # Selectăm două variabile pentru vizualizare
        col1, col2 = st.columns(2)
        with col1:
            x_axis = st.selectbox("Axează X", feature_cols, index=0, key="cluster_x")
        with col2:
            y_axis = st.selectbox("Axează Y", feature_cols, index=1, key="cluster_y")
        
        if x_axis == y_axis:
            st.warning("Alege două coloane diferite pentru X și Y!")
        else:
            # Creăm DataFrame-ul scalat
            df_scaled = pd.DataFrame(X_scaled, columns=feature_cols)
            df_scaled['Cluster'] = clusters
            
            # Creăm figura pentru clustere
            fig = go.Figure()
            
            # Definim o paletă de culori pentru clustere
            colors = px.colors.qualitative.Set1
            
            # Adăugăm punctele pentru fiecare cluster
            for cluster in range(n_clusters):
                cluster_data = df_scaled[df_scaled['Cluster'] == cluster]
                fig.add_trace(go.Scatter(
                    x=cluster_data[x_axis],
                    y=cluster_data[y_axis],
                    mode='markers',
                    name=f'Cluster {cluster}',
                    marker=dict(
                        size=8,
                        color=colors[cluster % len(colors)]
                    )
                ))
            
            # Adăugăm centroidele
            centroids = kmeans.cluster_centers_
            centroids_df = pd.DataFrame(centroids, columns=feature_cols)
            fig.add_trace(go.Scatter(
                x=centroids_df[x_axis],
                y=centroids_df[y_axis],
                mode='markers',
                name='Centroide',
                marker=dict(
                    size=12,
                    symbol='star',
                    color='black'
                )
            ))
            
            # Configurăm layout-ul
            fig.update_layout(
                title=dict(
                    text=f'Clustere în spațiul {x_axis} vs {y_axis} (date scalate)',
                    font=dict(size=16)
                ),
                xaxis_title=x_axis,
                yaxis_title=y_axis,
                showlegend=True,
                legend=dict(
                    title='Clustere',
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01
                )
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Adăugăm și un scatter plot cu datele originale pentru comparație
            fig_original = go.Figure()
            
            # Adăugăm punctele pentru fiecare cluster
            for cluster in range(n_clusters):
                cluster_data = df_clustered[df_clustered['Cluster'] == cluster]
                fig_original.add_trace(go.Scatter(
                    x=cluster_data[x_axis],
                    y=cluster_data[y_axis],
                    mode='markers',
                    name=f'Cluster {cluster}',
                    marker=dict(
                        size=8,
                        color=colors[cluster % len(colors)]
                    )
                ))
            
            # Configurăm layout-ul
            fig_original.update_layout(
                title=dict(
                    text=f'Clustere în spațiul {x_axis} vs {y_axis} (date originale)',
                    font=dict(size=16)
                ),
                xaxis_title=x_axis,
                yaxis_title=y_axis,
                showlegend=True,
                legend=dict(
                    title='Clustere',
                    yanchor="top",
                    y=0.99,
                    xanchor="left",
                    x=0.01
                )
            )
            
            st.plotly_chart(fig_original, use_container_width=True)
    else:
        st.warning("Selectați cel puțin două variabile pentru clusterizare!")