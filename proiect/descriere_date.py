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

#old load data, cu toate datele
@st.cache_data
def load_data():
    df = pd.read_csv('US_Accidents_March23.csv')

    df['Start_Time'] = pd.to_datetime(df['Start_Time'], format='mixed')
    df['End_Time'] = pd.to_datetime(df['End_Time'], format='mixed')

    df['Duration'] = (df['End_Time'] - df['Start_Time']).dt.total_seconds() / 60

    return df

st.title("📊 Analiza Exploratorie a Accidentelor Rutiere")

with st.sidebar:
    st.header("Meniu Principal")
    menu = st.radio(
        "Selectează secțiunea:",
        ["Analiză Generală", "Tratarea Valorilor Lipsă", "Identificarea Valorilor Extreme", "Grupări și Corelații"]
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
    st.dataframe(data_types, use_container_width=True)

    st.subheader("Statistici de bază")
    st.dataframe(filtered_df.describe(), use_container_width=True)

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
                mode_value = filtered_df[selected_col].mode()[0]
                df_replaced = filtered_df.copy()
                df_replaced[selected_col] = df_replaced[selected_col].fillna(mode_value)

                st.success(f"Pentru coloana categorică, valorile lipsă au fost înlocuite cu modul: {mode_value}")

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