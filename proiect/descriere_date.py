#Tema: Analiza exploratorie a datelor (prima a fost analiza descriptiva sau ceva de genul)
# Metode de tratare a valorilor lipsa (3 metode). Grupari pe date. Clusterizare dupa zona pentru harta
#Surprinderea valorilor extreme si tratarea lor. Sa schimbam graficul (de corelatie) cu seaborn. Buguri de fixat
#Analize de DSAD (Factorial, Clusterizare, ACP, Discriminanta)
#Sa folosim BoxPlot. Sa pot da click pe o coloana ca sa iasa boxplot/figura.
#Grupari de coloane
#Modularizare : definire de functie (ca e un singur fisier cu tot codul)
#Trebuia sa facem si meniu.
#Selectie pe coloane
#Sa folosim controale pe care le-am folosit in seminar2
#ne uitam de valori extreme (outlieri), ii pastram, ii eliminam? facem boxplot
#prelucram cu functii de grup, aplicam diverse functii de grup
#la tema folosim multe exemple cu loc si iloc (conditionat etc)
import streamlit as st
import pandas as pd
import numpy as np
import geopandas as gpd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go

# --------------------------
# Streamlit App Configuration
# --------------------------
st.set_page_config(page_title="US Accidents Analysis", layout="wide")


# --------------------------
# Modularizare - Definirea funcțiilor pentru organizarea codului
# --------------------------

# Funcția pentru încărcarea și preprocesarea datelor
@st.cache_data
def load_data():
    df = pd.read_csv('US_Accidents_March23.csv')

    # Convertire la datetime cu format='mixed' pentru a gestiona diverse formate
    df['Start_Time'] = pd.to_datetime(df['Start_Time'], format='mixed')
    df['End_Time'] = pd.to_datetime(df['End_Time'], format='mixed')

    # Calculează durata accidentului
    df['Duration'] = (df['End_Time'] - df['Start_Time']).dt.total_seconds() / 60

    return df


# Funcția pentru tratarea valorilor lipsă
def handle_missing_values(df, method, columns=None):
    if columns is None:
        columns = df.select_dtypes(include=['number']).columns.tolist()

    df_processed = df.copy()

    if method == "mean":
        for col in columns:
            if df[col].isna().any():
                df_processed[col] = df[col].fillna(df[col].mean())

    elif method == "median":
        for col in columns:
            if df[col].isna().any():
                df_processed[col] = df[col].fillna(df[col].median())

    elif method == "mode":
        for col in columns:
            if df[col].isna().any():
                df_processed[col] = df[col].fillna(df[col].mode()[0])

    elif method == "drop":
        if columns:
            df_processed = df_processed.dropna(subset=columns)
        else:
            df_processed = df_processed.dropna()

    elif method == "zero":
        for col in columns:
            if df[col].isna().any():
                df_processed[col] = df[col].fillna(0)

    elif method == "interpolate":
        for col in columns:
            if df[col].isna().any():
                df_processed[col] = df[col].interpolate(method='linear')

    return df_processed


# Funcție pentru detectarea outlier-ilor
def detect_outliers(df, column, method="iqr"):
    if method == "iqr":
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
        return outliers, lower_bound, upper_bound

    elif method == "zscore":
        z_scores = (df[column] - df[column].mean()) / df[column].std()
        outliers = df[abs(z_scores) > 3]
        return outliers, -3, 3

    return None, None, None


# Funcție pentru tratarea valorilor extreme
def handle_outliers(df, column, method="cap"):
    outliers, lower_bound, upper_bound = detect_outliers(df, column)
    df_processed = df.copy()

    if method == "cap":
        df_processed[column] = df_processed[column].clip(lower_bound, upper_bound)

    elif method == "remove":
        mask = (df_processed[column] >= lower_bound) & (df_processed[column] <= upper_bound)
        df_processed = df_processed[mask]

    elif method == "log":
        # Asigură-te că valorile sunt pozitive pentru transformarea logaritmică
        min_val = df_processed[column].min()
        if min_val <= 0:
            df_processed[column] = df_processed[column] - min_val + 1
        df_processed[column] = np.log(df_processed[column])

    return df_processed


# Funcție pentru aplicarea funcțiilor de grup
def apply_group_functions(df, group_by, agg_funcs, numeric_cols=None):
    if numeric_cols is None:
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()

    # Elimină coloanele care nu sunt dorite în agregare
    exclude_cols = ['ID']
    agg_cols = [col for col in numeric_cols if col not in exclude_cols]

    agg_dict = {col: agg_funcs for col in agg_cols}
    grouped_df = df.groupby(group_by).agg(agg_dict)

    return grouped_df


# --------------------------
# Titlul aplicației și descrierea setului de date
# --------------------------
st.title("📊 Analiza Exploratorie a Accidentelor Rutiere din SUA")

with st.expander("📋 Descrierea Setului de Date"):
    st.markdown("""
    ### Dicționar de Date

    Acest set de date conține informații despre accidentele rutiere din Statele Unite între 2016 și 2023. Iată ce reprezintă fiecare coloană:

    #### Identificare
    - **ID**: Identificator unic al înregistrării accidentului
    - **Source**: Sursa datelor brute privind accidentul

    #### Detalii Accident
    - **Severity**: Severitatea accidentului (1-4), unde 1 indică cel mai mic impact asupra traficului (întârziere scurtă) și 4 indică un impact semnificativ (întârziere lungă)
    - **Start_Time**: Ora de început a accidentului în fusul orar local
    - **End_Time**: Momentul în care impactul accidentului asupra fluxului de trafic a fost eliminat
    - **Distance(mi)**: Lungimea drumului afectat de accident în mile
    - **Description**: Descrierea accidentului furnizată de om

    #### Informații privind locația
    - **Start_Lat/Start_Lng**: Coordonate GPS ale punctului de început
    - **End_Lat/End_Lng**: Coordonate GPS ale punctului final
    - **Street**: Numele străzii
    - **City**: Numele orașului
    - **County**: Numele județului
    - **State**: Abrevierea statului
    - **Zipcode**: Codul poștal
    - **Country**: Țara (US)
    - **Timezone**: Fusul orar bazat pe locație (estic, central, etc.)

    #### Condiții Meteorologice
    - **Airport_Code**: Cea mai apropiată stație meteorologică bazată pe aeroport
    - **Weather_Timestamp**: Timpul observației meteorologice
    - **Temperature(F)**: Temperatura în Fahrenheit
    - **Wind_Chill(F)**: Senzația termică în Fahrenheit
    - **Humidity(%)**: Procentul de umiditate
    - **Pressure(in)**: Presiunea aerului în inch
    - **Visibility(mi)**: Vizibilitatea în mile
    - **Wind_Direction**: Direcția vântului
    - **Wind_Speed(mph)**: Viteza vântului în mph
    - **Precipitation(in)**: Cantitatea de precipitații în inch
    - **Weather_Condition**: Starea vremii (ploaie, zăpadă, etc.)

    #### Adnotări de Puncte de Interes (POI)
    Aceste câmpuri booleene indică prezența diverselor caracteristici în apropiere de accident:
    - **Amenity**, **Bump**, **Crossing**, **Give_Way**, **Junction**, **No_Exit**
    - **Railway**, **Roundabout**, **Station**, **Stop**
    - **Traffic_Calming**, **Traffic_Signal**, **Turning_Loop**

    #### Indicatori ai Momentului Zilei
    - **Sunrise_Sunset**: Zi sau noapte bazat pe răsărit/apus
    - **Civil_Twilight**: Zi sau noapte bazat pe crepusculul civil
    - **Nautical_Twilight**: Zi sau noapte bazat pe crepusculul nautic
    - **Astronomical_Twilight**: Zi sau noapte bazat pe crepusculul astronomic
    """)

# Încarcă datele
df = load_data()

# --------------------------
# Meniu Principal cu Selectie Tab
# --------------------------
menu_options = [
    "Prezentare Generală",
    "Tratarea Valorilor Lipsă",
    "Analiza Valorilor Extreme",
    "Analiză Temporală",
    "Grupări și Funcții de Grup",
    "Analiză Geospațială",
    "Analiză Avansată"
]

selected_menu = st.sidebar.radio("Selectați Secțiunea", menu_options)

# Opțiuni de filtrare în sidebar
with st.sidebar:
    st.header("Opțiuni de Filtrare")

    # Filtrare după ani
    min_year = df['Start_Time'].dt.year.min()
    max_year = df['Start_Time'].dt.year.max()
    selected_years = st.slider("Selectați Perioada",
                               min_value=int(min_year),
                               max_value=int(max_year),
                               value=(int(min_year), int(max_year)))

    # Filtrare după nivelul de severitate
    severity_levels = st.multiselect("Selectați Nivelurile de Severitate",
                                     options=sorted(df['Severity'].unique()),
                                     default=sorted(df['Severity'].unique()))

    # Filtrare după condiții meteo (top 5 cele mai frecvente)
    top_weather = df['Weather_Condition'].value_counts().head(5).index.tolist()
    weather_conditions = st.multiselect("Selectați Condițiile Meteo",
                                        options=df['Weather_Condition'].unique(),
                                        default=top_weather)

    # Selecție de coloane pentru afișare
    all_columns = df.columns.tolist()
    default_columns = ['ID', 'Start_Time', 'End_Time', 'Severity', 'City', 'State', 'Weather_Condition', 'Duration']
    selected_columns = st.multiselect("Selectați Coloanele pentru Afișare", all_columns, default=default_columns)

# --------------------------
# Filtrarea Datelor
# --------------------------
filtered_df = df[
    (df['Start_Time'].dt.year >= selected_years[0]) &
    (df['Start_Time'].dt.year <= selected_years[1]) &
    (df['Severity'].isin(severity_levels)) &
    (df['Weather_Condition'].isin(weather_conditions))
    ]

# --------------------------
# Secțiunea 1: Prezentare Generală
# --------------------------
if selected_menu == "Prezentare Generală":
    st.header("📊 Prezentare Generală a Datelor")

    # Metrici cheie
    st.markdown("### 📌 Metrici Cheie")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Accidente", f"{filtered_df.shape[0]:,}")
    with col2:
        st.metric("Severitate Medie", f"{filtered_df['Severity'].mean():.1f}/4")
    with col3:
        avg_duration = filtered_df['Duration'].mean()
        st.metric("Durată Medie", f"{avg_duration:.1f} minute")
    with col4:
        most_common_weather = filtered_df['Weather_Condition'].mode()[0]
        display_weather = most_common_weather if len(most_common_weather) < 15 else most_common_weather[:12] + "..."
        st.metric("Condiție Meteo Frecventă", display_weather)

    # Tab-uri pentru diferite vizualizări ale datelor
    tab1, tab2, tab3 = st.tabs(["Date Brute", "Statistici Descriptive", "Valori Lipsă"])

    with tab1:
        st.dataframe(filtered_df[selected_columns].head(1000),
                     use_container_width=True)

    with tab2:
        st.subheader("Statistici pentru Variabile Numerice")
        numeric_df = filtered_df.select_dtypes(include=['number'])
        stats_df = numeric_df.describe().round(2)
        st.dataframe(stats_df, use_container_width=True)

        st.subheader("Distribuția Variabilelor Categorice")
        cat_options = {
            'Severity': 'Nivelul de Severitate al Accidentului',
            'Weather_Condition': 'Condiții Meteo',
            'City': 'Oraș',
            'State': 'Stat',
            'Sunrise_Sunset': 'Momentul Zilei (Răsărit/Apus)'
        }

        cat_col = st.selectbox("Selectați Categoria pentru Analiză",
                               options=list(cat_options.keys()),
                               format_func=lambda x: cat_options[x])

        # Pentru o vizualizare mai bună, limitează la categoriile de top pentru câmpurile cu multe valori
        if cat_col in ['City', 'Weather_Condition']:
            counts = filtered_df[cat_col].value_counts().head(20).reset_index()
            title = f"Top 20 {cat_options[cat_col]} după Numărul de Accidente"
        else:
            counts = filtered_df[cat_col].value_counts().reset_index()
            title = f"Distribuția {cat_options[cat_col]}"

        fig = px.bar(counts, x=cat_col, y='count', title=title)

        # Rotește etichetele axei x pentru o lizibilitate mai bună, dacă este necesar
        if cat_col in ['City', 'Weather_Condition']:
            fig.update_layout(xaxis_tickangle=-45)

        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Distribuția Valorilor Lipsă")
        missing_values = filtered_df.isna().sum().reset_index()
        missing_values.columns = ['Coloană', 'Valori Lipsă']
        missing_values['Procent Lipsă'] = 100 * missing_values['Valori Lipsă'] / len(filtered_df)
        missing_values = missing_values.sort_values('Valori Lipsă', ascending=False)

        # Filtrează doar coloanele cu valori lipsă
        missing_values_nonzero = missing_values[missing_values['Valori Lipsă'] > 0]

        if not missing_values_nonzero.empty:
            fig = px.bar(missing_values_nonzero,
                         x='Coloană',
                         y='Valori Lipsă',
                         title="Distribuția Valorilor Lipsă",
                         hover_data=['Procent Lipsă'])
            st.plotly_chart(fig, use_container_width=True)

            st.dataframe(missing_values_nonzero, use_container_width=True)
        else:
            st.success("Nu există valori lipsă în setul de date filtrat!")

# --------------------------
# Secțiunea 2: Tratarea Valorilor Lipsă
# --------------------------
elif selected_menu == "Tratarea Valorilor Lipsă":
    st.header("🔍 Tratarea Valorilor Lipsă")

    st.markdown("""
    ### Metode de Tratare a Valorilor Lipsă

    Există mai multe metode de a trata valorile lipsă în setul de date. În această secțiune, puteți experimenta cu diferite strategii și observa efectele lor.
    """)

    # Afișează coloanele cu valori lipsă
    missing_values = df.isna().sum().reset_index()
    missing_values.columns = ['Coloană', 'Valori Lipsă']
    missing_values['Procent Lipsă'] = 100 * missing_values['Valori Lipsă'] / len(df)
    missing_values = missing_values.sort_values('Valori Lipsă', ascending=False)
    missing_values_nonzero = missing_values[missing_values['Valori Lipsă'] > 0]

    if not missing_values_nonzero.empty:
        st.subheader("Coloane cu Valori Lipsă")
        st.dataframe(missing_values_nonzero, use_container_width=True)

        # Selectează coloanele pentru tratarea valorilor lipsă
        missing_columns = missing_values_nonzero['Coloană'].tolist()
        selected_missing_columns = st.multiselect(
            "Selectați coloanele pentru a trata valorile lipsă",
            options=missing_columns,
            default=missing_columns[:3] if len(missing_columns) >= 3 else missing_columns
        )

        # Selectează metoda de tratare
        method_options = {
            "mean": "Înlocuire cu Media",
            "median": "Înlocuire cu Mediana",
            "mode": "Înlocuire cu Modul (valoarea cea mai frecventă)",
            "zero": "Înlocuire cu Zero",
            "drop": "Eliminarea Rândurilor cu Valori Lipsă",
            "interpolate": "Interpolare Liniară"
        }

        method = st.radio(
            "Selectați metoda de tratare a valorilor lipsă",
            options=list(method_options.keys()),
            format_func=lambda x: method_options[x]
        )

        if st.button("Aplică Metoda"):
            if selected_missing_columns:
                with st.spinner("Se aplică metoda de tratare a valorilor lipsă..."):
                    # Aplică metoda selecționată doar pe setul filtrat, nu pe tot setul de date
                    processed_df = handle_missing_values(filtered_df, method, selected_missing_columns)

                    # Compară valorile înainte și după
                    st.success(f"Metoda '{method_options[method]}' a fost aplicată cu succes!")

                    # Verifică dacă mai există valori lipsă
                    remaining_missing = processed_df[selected_missing_columns].isna().sum()

                    # Afișează statistici pentru coloanele procesate
                    st.subheader("Statistici înainte și după procesare")

                    for col in selected_missing_columns:
                        if col in filtered_df.select_dtypes(include=['number']).columns:
                            col1, col2 = st.columns(2)

                            with col1:
                                st.markdown(f"**{col} - Înainte**")
                                st.dataframe(filtered_df[col].describe().to_frame(), use_container_width=True)

                            with col2:
                                st.markdown(f"**{col} - După**")
                                st.dataframe(processed_df[col].describe().to_frame(), use_container_width=True)

                            # Afișează histograme pentru a compara distribuțiile
                            fig, ax = plt.subplots(1, 2, figsize=(12, 5))

                            # Histogram înainte
                            sns.histplot(filtered_df[col].dropna(), kde=True, ax=ax[0])
                            ax[0].set_title(f"{col} - Înainte")

                            # Histogram după
                            sns.histplot(processed_df[col], kde=True, ax=ax[1])
                            ax[1].set_title(f"{col} - După")

                            st.pyplot(fig)

                    if remaining_missing.sum() > 0:
                        st.warning("Unele valori lipsă nu au fost tratate. Verificați datele.")
                        st.dataframe(remaining_missing[remaining_missing > 0].to_frame("Valori Lipsă Rămase"))
                    else:
                        st.success("Toate valorile lipsă au fost tratate cu succes pentru coloanele selectate!")
            else:
                st.error("Selectați cel puțin o coloană pentru a aplica metoda de tratare a valorilor lipsă.")
    else:
        st.success("Nu există valori lipsă în setul de date filtrat!")

# --------------------------
# Secțiunea 3: Analiza Valorilor Extreme
# --------------------------
elif selected_menu == "Analiza Valorilor Extreme":
    st.header("📏 Analiza Valorilor Extreme (Outliers)")

    st.markdown("""
    ### Detectarea și Tratarea Valorilor Extreme

    Valorile extreme (outliers) pot influența semnificativ rezultatele analizelor statistice. În această secțiune, puteți:
    1. Detecta valorile extreme folosind metoda IQR (Intervalul Intercuartilic) sau Z-score
    2. Vizualiza distribuțiile folosind boxplot-uri
    3. Trata valorile extreme prin diverse metode
    """)

    # Selectează variabilele numerice pentru analiză
    numeric_columns = df.select_dtypes(include=['number']).columns.tolist()
    # Exclude ID-ul și alte coloane care nu ar trebui analizate pentru outliers
    exclude_cols = ['ID']
    numeric_columns = [col for col in numeric_columns if col not in exclude_cols]

    # Selectează variabila pentru analiză
    selected_var = st.selectbox(
        "Selectați variabila pentru analiza valorilor extreme",
        options=numeric_columns,
        index=0
    )

    col1, col2 = st.columns(2)

    with col1:
        # Metoda de detectare a valorilor extreme
        detection_method = st.radio(
            "Metoda de detectare",
            options=["iqr", "zscore"],
            format_func=lambda x: "IQR (Intervalul Intercuartilic)" if x == "iqr" else "Z-score"
        )

    with col2:
        # Metoda de tratare a valorilor extreme
        treatment_method = st.radio(
            "Metoda de tratare",
            options=["cap", "remove", "log"],
            format_func=lambda x: {
                "cap": "Limitare (Capping)",
                "remove": "Eliminare",
                "log": "Transformare Logaritmică"
            }[x]
        )

    # Detectează valorile extreme
    outliers, lower_bound, upper_bound = detect_outliers(filtered_df, selected_var, detection_method)

    if outliers is not None and len(outliers) > 0:
        st.subheader(f"Valori Extreme pentru {selected_var}")

        outlier_percentage = (len(outliers) / len(filtered_df)) * 100

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Număr de Outlieri", f"{len(outliers):,}")
        with col2:
            st.metric("Procent din Date", f"{outlier_percentage:.2f}%")
        with col3:
            if detection_method == "iqr":
                st.metric("Interval Valid", f"[{lower_bound:.2f}, {upper_bound:.2f}]")
            else:
                st.metric("Z-score Limită", f"[-3, 3]")

        # Vizualizează distribuția și valorile extreme cu boxplot
        st.subheader("Vizualizare BoxPlot")
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(x=filtered_df[selected_var], ax=ax)
        ax.set_title(f"BoxPlot pentru {selected_var}")
        st.pyplot(fig)

        # Vizualizează distribuția mai detaliată
        st.subheader("Distribuția și Valorile Extreme")
        fig = px.histogram(filtered_df, x=selected_var, marginal="box",
                           title=f"Distribuția pentru {selected_var}")
        st.plotly_chart(fig, use_container_width=True)

        # Tratează valorile extreme
        if st.button("Aplică Tratamentul pentru Valorile Extreme"):
            with st.spinner("Se aplică tratamentul pentru valorile extreme..."):
                processed_df = handle_outliers(filtered_df, selected_var, treatment_method)

                # Compară statisticile înainte și după
                st.success(f"Metoda de tratare '{treatment_method}' a fost aplicată cu succes!")

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown(f"**Statistici înainte de tratare**")
                    st.dataframe(filtered_df[selected_var].describe().to_frame(), use_container_width=True)

                with col2:
                    st.markdown(f"**Statistici după tratare**")
                    st.dataframe(processed_df[selected_var].describe().to_frame(), use_container_width=True)

                # Vizualizează distribuția după tratare
                fig, ax = plt.subplots(1, 2, figsize=(15, 6))

                # BoxPlot înainte
                sns.boxplot(x=filtered_df[selected_var], ax=ax[0])
                ax[0].set_title(f"{selected_var} - Înainte")

                # BoxPlot după
                sns.boxplot(x=processed_df[selected_var], ax=ax[1])
                ax[1].set_title(f"{selected_var} - După")

                st.pyplot(fig)

                # Histograme înainte și după
                fig = px.histogram(
                    pd.DataFrame({
                        'Înainte': filtered_df[selected_var],
                        'După': processed_df[selected_var]
                    }).melt(),
                    x="value", color="variable",
                    marginal="box",
                    title=f"Comparație Distribuție {selected_var} - Înainte vs După"
                )
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(f"Nu s-au detectat valori extreme pentru {selected_var} folosind metoda {detection_method}.")

# --------------------------
# Secțiunea 4: Analiză Temporală
# --------------------------
elif selected_menu == "Analiză Temporală":
    st.header("⏱️ Analiză Temporală")

    st.markdown("""
    ### Explorarea Modelelor Temporale

    Această secțiune analizează distribuția accidentelor în funcție de timp: ore, zile, luni, ani.
    """)

    # Opțiuni de grupare temporală
    temporal_options = {
        "hour": "Oră din Zi",
        "day": "Zi din Săptămână",
        "month": "Lună din An",
        "year": "An",
        "season": "Sezon"
    }

    temporal_group = st.radio(
        "Selectați gruparea temporală",
        options=list(temporal_options.keys()),
        format_func=lambda x: temporal_options[x]
    )

    # Pregătește datele pentru gruparea selectată
    if temporal_group == "hour":
        temporal_df = filtered_df.copy()
        temporal_df['temporal_group'] = temporal_df['Start_Time'].dt.hour
        group_name = "Ora"
        # Creează etichete formatate pentru ore (e.g., "01:00", "02:00")
        hour_labels = {hr: f"{hr:02d}:00" for hr in range(24)}
        temporal_df['group_label'] = temporal_df['temporal_group'].map(hour_labels)
        group_order = [hour_labels[hr] for hr in range(24)]

    elif temporal_group == "day":
        temporal_df = filtered_df.copy()
        temporal_df['temporal_group'] = temporal_df['Start_Time'].dt.dayofweek
        group_name = "Ziua Săptămânii"
        # Mapare zile ale săptămânii
        day_labels = {
            0: "Luni", 1: "Marți", 2: "Miercuri", 3: "Joi",
            4: "Vineri", 5: "Sâmbătă", 6: "Duminică"
        }
        temporal_df['group_label'] = temporal_df['temporal_group'].map(day_labels)
        group_order = [day_labels[day] for day in range(7)]

    elif temporal_group == "month":
        temporal_df = filtered_df.copy()
        temporal_df['temporal_group'] = temporal_df['Start_Time'].dt.month
        group_name = "Luna"
        # Mapare luni
        month_labels = {
            1: "Ianuarie", 2: "Februarie", 3: "Martie", 4: "Aprilie",
            5: "Mai", 6: "Iunie", 7: "Iulie", 8: "August",
            9: "Septembrie", 10: "Octombrie", 11: "Noiembrie", 12: "Decembrie"
        }
        temporal_df['group_label'] = temporal_df['Start_Time'].dt.month.map(month_labels)
        group_order = [month_labels[month] for month in range(1, 13)]

    elif temporal_group == "year":
        temporal_df = filtered_df.copy()
        temporal_df['temporal_group'] = temporal_df['Start_Time'].dt.year
        temporal_df['group_label'] = temporal_df['temporal_group'].astype(str)
        group_name = "Anul"
        group_order = sorted(temporal_df['group_label'].unique())

    elif temporal_group == "season":
        temporal_df = filtered_df.copy()
        # Definirea sezoanelor (pentru emisfera nordică)
        # Iarnă: Dec, Ian, Feb; Primăvară: Mar, Apr, Mai; Vară: Iun, Iul, Aug; Toamnă: Sep, Oct, Noi
        month_to_season = {
            12: "Iarnă", 1: "Iarnă", 2: "Iarnă",
            3: "Primăvară", 4: "Primăvară", 5: "Primăvară",
            6: "Vară", 7: "Vară", 8: "Vară",
            9: "Toamnă", 10: "Toamnă", 11: "Toamnă"
        }
        temporal_df['temporal_group'] = temporal_df['Start_Time'].dt.month.map(month_to_season)
        temporal_df['group_label'] = temporal_df['temporal_group']
        group_name = "Sezonul"
        group_order = ["Primăvară", "Vară", "Toamnă", "Iarnă"]

    # Analiză temporală - opțiuni de vizualizare
    viz_type = st.radio(
        "Alegeți tipul de vizualizare",
        options=["Număr de accidente", "Severitate medie", "Durată medie"]
    )

    if viz_type == "Număr de accidente":
        # Numără accidentele pe grupă temporală
        temp_counts = temporal_df.groupby('group_label').size().reset_index(name='count')

        # Dacă există o ordine specificată, sortează datele conform acesteia
        if group_order:
            temp_counts['group_label'] = pd.Categorical(temp_counts['group_label'], categories=group_order,
                                                        ordered=True)
            temp_counts = temp_counts.sort_values('group_label')

        fig = px.bar(
            temp_counts,
            x='group_label',
            y='count',
            title=f'Număr de Accidente per {group_name}',
            labels={'group_label': group_name, 'count': 'Număr de Accidente'}
        )

    elif viz_type == "Severitate medie":
        # Calculează severitatea medie pe grupă temporală
        temp_severity = temporal_df.groupby('group_label')['Severity'].mean().reset_index()

        # Dacă există o ordine specificată, sortează datele conform acesteia
        if group_order:
            temp_severity['group_label'] = pd.Categorical(temp_severity['group_label'], categories=group_order,
                                                          ordered=True)
            temp_severity = temp_severity.sort_values('group_label')

        fig = px.bar(
            temp_severity,
            x='group_label',
            y='Severity',
            title=f'Severitate Medie per {group_name}',
            labels={'group_label': group_name, 'Severity': 'Severitate Medie'},
            color='Severity',
            color_continuous_scale='Reds'
        )

    else:  # Durată medie
        # Calculează durata medie pe grupă temporală
        temp_duration = temporal_df.groupby('group_label')['Duration'].mean().reset_index()

        # Dacă există o ordine specificată, sortează datele conform acesteia
        if group_order:
            temp_duration['group_label'] = pd.Categorical(temp_duration['group_label'], categories=group_order,
                                                          ordered=True)
            temp_duration = temp_duration.sort_values('group_label')

        fig = px.bar(
            temp_duration,
            x='group_label',
            y='Duration',
            title=f'Durată Medie a Accidentelor per {group_name}',
            labels={'group_label': group_name, 'Duration': 'Durată Medie (minute)'},
            color='Duration',
            color_continuous_scale='Blues'
        )

    # Afișează graficul
    st.plotly_chart(fig, use_container_width=True)

    # Analiza tendințelor temporale (doar pentru lună și an)
    if temporal_group in ["month", "year"]:
        st.subheader(f"Tendințe Temporale de-a Lungul {temporal_options[temporal_group]}elor")

        # Crează o serie de timp
        if temporal_group == "month":
            temporal_df['date'] = pd.to_datetime(temporal_df['Start_Time'].dt.strftime('%Y-%m-01'))
        else:  # year
            temporal_df['date'] = pd.to_datetime(temporal_df['Start_Time'].dt.strftime('%Y-01-01'))

        # Grupează după data (an-lună sau an)
        time_series = temporal_df.groupby('date').agg({
            'ID': 'count',
            'Severity': 'mean',
            'Duration': 'mean'
        }).reset_index()

        time_series.columns = ['Data', 'Număr Accidente', 'Severitate Medie', 'Durată Medie']

        # Crează un grafic de serie temporală
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=time_series['Data'],
            y=time_series['Număr Accidente'],
            mode='lines+markers',
            name='Număr Accidente'
        ))

        fig.update_layout(
            title=f'Tendințe în Numărul de Accidente de-a Lungul Timpului',
            xaxis_title='Data',
            yaxis_title='Număr de Accidente'
        )

        st.plotly_chart(fig, use_container_width=True)

        # Opțiune pentru a vizualiza și severitatea și durata
        if st.checkbox("Afișează și tendințe pentru severitate și durată"):
            col1, col2 = st.columns(2)

            with col1:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=time_series['Data'],
                    y=time_series['Severitate Medie'],
                    mode='lines+markers',
                    name='Severitate Medie',
                    line=dict(color='red')
                ))

                fig.update_layout(
                    title='Tendințe în Severitatea Accidentelor',
                    xaxis_title='Data',
                    yaxis_title='Severitate Medie'
                )

                st.plotly_chart(fig, use_container_width=True)

            with col2:
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=time_series['Data'],
                    y=time_series['Durată Medie'],
                    mode='lines+markers',
                    name='Durată Medie',
                    line=dict(color='blue')
                ))

                fig.update_layout(
                    title='Tendințe în Durata Accidentelor',
                    xaxis_title='Data',
                    yaxis_title='Durată Medie (minute)'
                )

                st.plotly_chart(fig, use_container_width=True)

    # Heatmap cu două dimensiuni temporale
    st.subheader("Heatmap Temporal")

    # Selectăm dimensiunile pentru heatmap
    temp_dim1 = st.selectbox("Prima dimensiune temporală",
                             options=["hour", "day", "month", "season"],
                             format_func=lambda x: temporal_options[x])

    temp_dim2 = st.selectbox("A doua dimensiune temporală",
                             options=["day", "month", "season", "year"],
                             format_func=lambda x: temporal_options[x],
                             index=2)

    # Definim mapările de etichete pentru fiecare dimensiune
    hour_labels = {hr: f"{hr:02d}:00" for hr in range(24)}
    day_labels = {0: "Luni", 1: "Marți", 2: "Miercuri", 3: "Joi", 4: "Vineri", 5: "Sâmbătă", 6: "Duminică"}
    month_labels = {1: "Ian", 2: "Feb", 3: "Mar", 4: "Apr", 5: "Mai", 6: "Iun",
                    7: "Iul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Noi", 12: "Dec"}
    season_labels = {
        "Primăvară": "Primăvară", "Vară": "Vară", "Toamnă": "Toamnă", "Iarnă": "Iarnă"
    }


    # Funcție pentru a obține dimensiunile temporale
    def get_temporal_dimension(df, dim):
        if dim == "hour":
            return df['Start_Time'].dt.hour.map(hour_labels)
        elif dim == "day":
            return df['Start_Time'].dt.dayofweek.map(day_labels)
        elif dim == "month":
            return df['Start_Time'].dt.month.map(month_labels)
        elif dim == "season":
            return df['Start_Time'].dt.month.map({
                12: "Iarnă", 1: "Iarnă", 2: "Iarnă",
                3: "Primăvară", 4: "Primăvară", 5: "Primăvară",
                6: "Vară", 7: "Vară", 8: "Vară",
                9: "Toamnă", 10: "Toamnă", 11: "Toamnă"
            })
        elif dim == "year":
            return df['Start_Time'].dt.year.astype(str)


    # Calculează valorile pentru heatmap
    heatmap_df = filtered_df.copy()
    heatmap_df['dim1'] = get_temporal_dimension(heatmap_df, temp_dim1)
    heatmap_df['dim2'] = get_temporal_dimension(heatmap_df, temp_dim2)

    # Grupează datele pentru heatmap
    heatmap_data = heatmap_df.groupby(['dim1', 'dim2']).size().reset_index(name='count')
    heatmap_pivot = heatmap_data.pivot(index='dim1', columns='dim2', values='count').fillna(0)

    # Afișează heatmap-ul
    fig = px.imshow(
        heatmap_pivot,
        labels=dict(x=temporal_options[temp_dim2], y=temporal_options[temp_dim1], color="Număr de Accidente"),
        x=heatmap_pivot.columns,
        y=heatmap_pivot.index,
        title=f'Heatmap: {temporal_options[temp_dim1]} vs {temporal_options[temp_dim2]}'
    )

    st.plotly_chart(fig, use_container_width=True)

# --------------------------
# Secțiunea 5: Grupări și Funcții de Grup
# --------------------------
elif selected_menu == "Grupări și Funcții de Grup":
    st.header("🔄 Grupări și Funcții de Grup")

    st.markdown("""
    ### Aplicarea Funcțiilor de Grup

    Această secțiune permite gruparea datelor după diferite criterii și aplicarea diverselor funcții de agregare.
    """)

    # Selectează variabilele pentru grupare
    group_options = {
        "State": "Stat",
        "City": "Oraș",
        "Weather_Condition": "Condiție Meteo",
        "Severity": "Severitate",
        "Sunrise_Sunset": "Moment al Zilei",
    }

    col1, col2 = st.columns(2)

    with col1:
        # Variabile de grupare
        group_vars = st.multiselect(
            "Selectați variabilele pentru grupare",
            options=list(group_options.keys()),
            default=["State"],
            format_func=lambda x: group_options[x]
        )

    with col2:
        # Funcții de agregare
        agg_functions = st.multiselect(
            "Selectați funcțiile de agregare",
            options=["count", "mean", "sum", "min", "max", "std", "var"],
            default=["count", "mean"],
            format_func=lambda x: {
                "count": "Număr", "mean": "Media", "sum": "Suma",
                "min": "Minim", "max": "Maxim", "std": "Deviația Standard", "var": "Varianța"
            }[x]
        )

    # Selectează variabilele pentru agregare
    numeric_cols = filtered_df.select_dtypes(include=['number']).columns.tolist()
    exclude_cols = ['ID']
    numeric_cols = [col for col in numeric_cols if col not in exclude_cols]

    agg_vars = st.multiselect(
        "Selectați variabilele pentru agregare",
        options=numeric_cols,
        default=["Severity", "Duration"] if "Severity" in numeric_cols and "Duration" in numeric_cols else numeric_cols[
                                                                                                           :2]
    )

    if st.button("Aplică Gruparea"):
        if group_vars and agg_functions and agg_vars:
            with st.spinner("Se aplică funcțiile de grup..."):
                # Creează un dicționar pentru funcțiile de agregare
                agg_dict = {var: agg_functions for var in agg_vars}

                # Aplică gruparea și funcțiile de agregare
                grouped_df = filtered_df.groupby(group_vars).agg(agg_dict)

                # Resetează indexul pentru o vizualizare mai bună
                grouped_df = grouped_df.reset_index()

                # Afișează rezultatele
                st.subheader("Rezultatele Grupării")
                st.dataframe(grouped_df, use_container_width=True)

                # Vizualizări pentru rezultatele grupării
                if len(group_vars) == 1 and 'count' in agg_functions and agg_vars:
                    st.subheader(f"Top 20 {group_options[group_vars[0]]} după Număr de Accidente")

                    # Extrage coloana count
                    count_col = f"{agg_vars[0]}_count"

                    # Sortează și limitează la top 20
                    top_groups = grouped_df.sort_values(count_col, ascending=False).head(20)

                    fig = px.bar(
                        top_groups,
                        x=group_vars[0],
                        y=count_col,
                        title=f'Top 20 {group_options[group_vars[0]]} după Număr de Accidente',
                        color=count_col,
                        labels={count_col: 'Număr de Accidente', group_vars[0]: group_options[group_vars[0]]}
                    )

                    # Rotește etichetele pentru o mai bună lizibilitate
                    if group_vars[0] in ['City', 'Weather_Condition']:
                        fig.update_layout(xaxis_tickangle=-45)

                    st.plotly_chart(fig, use_container_width=True)

                # Vizualizare pentru medii dacă sunt disponibile
                if len(group_vars) == 1 and 'mean' in agg_functions and len(agg_vars) >= 2:
                    st.subheader(f"Comparație Medii pentru Top 10 {group_options[group_vars[0]]}")

                    # Extrage coloanele pentru medii
                    mean_cols = [f"{var}_mean" for var in agg_vars]

                    # Sortează după prima coloană de medie și limitează la top 10
                    top_means = grouped_df.sort_values(mean_cols[0], ascending=False).head(10)

                    # Pregătește datele pentru vizualizare
                    plot_data = top_means.melt(
                        id_vars=group_vars,
                        value_vars=mean_cols,
                        var_name="Variabilă",
                        value_name="Valoare Medie"
                    )

                    # Curăță numele variabilelor pentru afișare
                    plot_data['Variabilă'] = plot_data['Variabilă'].apply(lambda x: x.split('_mean')[0])

                    fig = px.bar(
                        plot_data,
                        x=group_vars[0],
                        y="Valoare Medie",
                        color="Variabilă",
                        barmode="group",
                        title=f'Comparație Medii pentru Top 10 {group_options[group_vars[0]]}',
                        labels={group_vars[0]: group_options[group_vars[0]]}
                    )

                    # Rotește etichetele pentru o mai bună lizibilitate
                    if group_vars[0] in ['City', 'Weather_Condition']:
                        fig.update_layout(xaxis_tickangle=-45)

                    st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning(
                "Vă rugăm să selectați cel puțin o variabilă pentru grupare, o funcție de agregare și o variabilă pentru agregare.")

# --------------------------
# Secțiunea 6: Analiză Geospațială
# --------------------------
elif selected_menu == "Analiză Geospațială":
    st.header("🌍 Analiză Geospațială")

    st.markdown("""
    ### Analiza Distribuției Geografice a Accidentelor

    Această secțiune permite vizualizarea distribuției geografice a accidentelor și analizarea tiparelor spațiale.
    """)

    # Verifică dacă avem date de geolocalizare valide
    if 'Start_Lat' in filtered_df.columns and 'Start_Lng' in filtered_df.columns:
        # Elimină rândurile cu coordonate geografice lipsă
        geo_df = filtered_df.dropna(subset=['Start_Lat', 'Start_Lng'])

        if not geo_df.empty:
            st.subheader("Hartă de Densitate a Accidentelor")

            # Pentru performanță, eșantionăm datele dacă sunt prea multe
            sample_size = min(10000, len(geo_df))
            if len(geo_df) > sample_size:
                st.info(
                    f"Pentru performanță, se afișează un eșantion aleatoriu de {sample_size} accidente din totalul de {len(geo_df)}.")
                map_data = geo_df.sample(sample_size)
            else:
                map_data = geo_df

            # Selectează variabila pentru colorare
            color_var = st.selectbox(
                "Colorează punctele după",
                options=["Severity", "Duration", "Weather_Condition"],
                format_func=lambda x: {
                    "Severity": "Severitate",
                    "Duration": "Durată",
                    "Weather_Condition": "Condiție Meteo"
                }[x]
            )

            # Creează harta
            if color_var in ["Severity", "Duration"]:
                fig = px.scatter_mapbox(
                    map_data,
                    lat='Start_Lat',
                    lon='Start_Lng',
                    color=color_var,
                    size=color_var if color_var == "Severity" else None,
                    color_continuous_scale="Viridis" if color_var == "Duration" else "Reds",
                    size_max=10,
                    zoom=3,
                    mapbox_style="carto-positron",
                    title="Distribuția Geografică a Accidentelor",
                    hover_data=['City', 'Weather_Condition', 'Severity', 'Duration']
                )
            else:  # Weather_Condition
                fig = px.scatter_mapbox(
                    map_data,
                    lat='Start_Lat',
                    lon='Start_Lng',
                    color='Weather_Condition',
                    size="Severity",
                    size_max=10,
                    zoom=3,
                    mapbox_style="carto-positron",
                    title="Distribuția Geografică a Accidentelor",
                    hover_data=['City', 'Weather_Condition', 'Severity', 'Duration']
                )

            fig.update_layout(margin={"r": 0, "t": 40, "l": 0, "b": 0})
            st.plotly_chart(fig, use_container_width=True)

            # Agregare geografică la nivel de stat
            st.subheader("Analiză la Nivel de Stat")

            # Grupează datele după stat
            state_data = filtered_df.groupby('State').agg({
                'ID': 'count',
                'Severity': 'mean',
                'Duration': 'mean'
            }).reset_index()

            state_data.columns = ['State', 'Număr Accidente', 'Severitate Medie', 'Durată Medie']
            state_data = state_data.sort_values('Număr Accidente', ascending=False)

            # Afișează statistici la nivel de stat
            st.dataframe(state_data, use_container_width=True)

            # Vizualizează top 10 state după numărul de accidente
            st.subheader("Top 10 State după Numărul de Accidente")

            fig = px.bar(
                state_data.head(10),
                x='State',
                y='Număr Accidente',
                color='Severitate Medie',
                color_continuous_scale='Reds',
                title='Top 10 State după Numărul de Accidente',
                labels={'State': 'Stat', 'Număr Accidente': 'Număr de Accidente'}
            )

            st.plotly_chart(fig, use_container_width=True)

            # Comparație între state pentru severitate și durată
            st.subheader("Comparație între State: Severitate vs Durată")

            fig = px.scatter(
                state_data,
                x='Severitate Medie',
                y='Durată Medie',
                size='Număr Accidente',
                color='Număr Accidente',
                hover_name='State',
                log_x=False,
                log_y=False,
                size_max=60,
                title='Severitate vs Durată pe State (dimensiunea = număr de accidente)'
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Nu există date geografice valide în setul de date filtrat.")
    else:
        st.error("Lipsesc coloanele de coordonate geografice din setul de date.")

# --------------------------
# Secțiunea 7: Analiză Avansată
# --------------------------
elif selected_menu == "Analiză Avansată":
    st.header("🔍 Analiză Avansată")

    st.markdown("""
    ### Analize și Vizualizări Avansate

    Această secțiune oferă analize și vizualizări mai avansate pentru setul de date.
    """)

    # Opțiuni pentru diferite tipuri de analiză
    analysis_type = st.radio(
        "Selectați tipul de analiză",
        options=["Matrice de Corelație", "Analiză BoxPlot", "Distribuții Bivariate", "Analiza POI"]
    )

    if analysis_type == "Matrice de Corelație":
        st.subheader("Matrice de Corelație pentru Variabile Numerice")

        # Selectează variabilele numerice pentru analiza de corelație
        numeric_cols = filtered_df.select_dtypes(include=['number']).columns.tolist()
        exclude_cols = ['ID']
        corr_cols = [col for col in numeric_cols if col not in exclude_cols]

        selected_corr_cols = st.multiselect(
            "Selectați variabilele pentru analiza de corelație",
            options=corr_cols,
            default=corr_cols[:6] if len(corr_cols) > 6 else corr_cols
        )

        if selected_corr_cols:
            # Calculează matricea de corelație
            corr_df = filtered_df[selected_corr_cols].corr()

            # Selectează metoda de vizualizare (Plotly sau Seaborn)
            viz_method = st.radio(
                "Metodă de vizualizare",
                options=["Plotly", "Seaborn"]
            )

            if viz_method == "Plotly":
                fig = px.imshow(
                    corr_df,
                    text_auto=True,
                    color_continuous_scale='RdBu_r',
                    title="Matrice de Corelație",
                    labels=dict(x="Variabile", y="Variabile", color="Corelație")
                )

                st.plotly_chart(fig, use_container_width=True)
            else:  # Seaborn
                st.subheader("Matrice de Corelație cu Seaborn")

                fig, ax = plt.subplots(figsize=(10, 8))
                sns.heatmap(corr_df, annot=True, cmap='coolwarm', linewidths=0.5, ax=ax)
                st.pyplot(fig)

            # Afișarea celor mai puternice corelații
            st.subheader("Cele Mai Puternice Corelații")

            # Transformă matricea într-un DataFrame pentru afișare
            corr_pairs = corr_df.unstack().sort_values(ascending=False)

            # Elimină perechile diagonale (corelații cu ele însele)
            corr_pairs = corr_pairs[corr_pairs < 1.0]

            # Afișează top 10 corelații pozitive și negative
            col1, col2 = st.columns(2)

            with col1:
                st.markdown("**Top 10 Corelații Pozitive**")
                st.dataframe(corr_pairs.head(10).reset_index())

            with col2:
                st.markdown("**Top 10 Corelații Negative**")
                st.dataframe(corr_pairs.tail(10).reset_index())
        else:
            st.warning("Selectați cel puțin o variabilă pentru analiza de corelație.")

    elif analysis_type == "Analiză BoxPlot":
        st.subheader("Analiză BoxPlot pentru Variabile Numerice")

        # Selectează variabilele numerice pentru analiza BoxPlot
        numeric_cols = filtered_df.select_dtypes(include=['number']).columns.tolist()
        exclude_cols = ['ID']
        box_cols = [col for col in numeric_cols if col not in exclude_cols]

        col1, col2 = st.columns(2)

        with col1:
            # Variabile pentru axa x (grupare)
            box_var_x = st.selectbox(
                "Selectați variabila categorică pentru grupare",
                options=["Severity", "Weather_Condition", "Sunrise_Sunset", "State"],
                format_func=lambda x: {
                    "Severity": "Severitate",
                    "Weather_Condition": "Condiție Meteo",
                    "Sunrise_Sunset": "Moment al Zilei",
                    "State": "Stat"
                }[x]
            )

        with col2:
            # Variabile pentru axa y (valori)
            box_var_y = st.selectbox(
                "Selectați variabila numerică pentru BoxPlot",
                options=box_cols,
                index=box_cols.index("Duration") if "Duration" in box_cols else 0
            )

        # Opțiune pentru sortarea valorilor
        sort_values = st.checkbox("Sortează valorile", value=True)

        # Limitarea numărului de categorii (pentru variabile cu multe categorii)
        if box_var_x in ["Weather_Condition", "State", "City"]:
            n_categories = st.slider(
                f"Număr de categorii pentru {box_var_x}",
                min_value=5,
                max_value=30,
                value=10
            )

            # Găsește cele mai frecvente categorii
            top_categories = filtered_df[box_var_x].value_counts().head(n_categories).index.tolist()
            plot_df = filtered_df[filtered_df[box_var_x].isin(top_categories)]

            if sort_values:
                # Sortează categoriile după valoarea mediană a variabilei numerice
                category_order = plot_df.groupby(box_var_x)[box_var_y].median().sort_values().index.tolist()
            else:
                category_order = top_categories
        else:
            plot_df = filtered_df

            if sort_values and box_var_x != "Severity":
                # Sortează categoriile după valoarea mediană a variabilei numerice
                category_order = plot_df.groupby(box_var_x)[box_var_y].median().sort_values().index.tolist()
            else:
                category_order = None

        # Creează BoxPlot
        st.subheader(f"BoxPlot: {box_var_y} grupat după {box_var_x}")

        fig = px.box(
            plot_df,
            x=box_var_x,
            y=box_var_y,
            color=box_var_x,
            category_orders={box_var_x: category_order} if category_order else None,
            title=f"BoxPlot: {box_var_y} grupat după {box_var_x}",
            labels={box_var_x: box_var_x, box_var_y: box_var_y}
        )

        # Rotește etichetele pentru o mai bună lizibilitate
        if box_var_x in ["Weather_Condition", "City"]:
            fig.update_layout(xaxis_tickangle=-45)

        st.plotly_chart(fig, use_container_width=True)

        # Adaugă o informație despre valorile extreme
        st.info("""
        **Interpretarea BoxPlot-ului:**
        - Cutia centrală reprezintă intervalul între cuartila 1 (Q1) și cuartila 3 (Q3)
        - Linia din interiorul cutiei reprezintă mediana (Q2)
        - Mustățile se extind până la valori aflate la 1.5 * IQR (Intervalul Intercuartilic) de la marginile cutiei
        - Punctele individuale reprezintă valorile extreme (outlieri)
        """)

        # Afișează statistici pentru fiecare grupă
        if st.checkbox("Afișează statistici pentru fiecare grupă"):
            stats = plot_df.groupby(box_var_x)[box_var_y].describe().reset_index()
            st.dataframe(stats, use_container_width=True)

    elif analysis_type == "Distribuții Bivariate":
        st.subheader("Analiză Bivariată pentru Variabile Numerice")

        # Selectează variabilele numerice pentru analiza bivariată
        numeric_cols = filtered_df.select_dtypes(include=['number']).columns.tolist()
        exclude_cols = ['ID']
        bivar_cols = [col for col in numeric_cols if col not in exclude_cols]

        col1, col2, col3 = st.columns(3)

        with col1:
            # Variabila pentru axa x
            bivar_x = st.selectbox(
                "Variabila pentru axa X",
                options=bivar_cols,
                index=bivar_cols.index("Temperature(F)") if "Temperature(F)" in bivar_cols else 0
            )

        with col2:
            # Variabila pentru axa y
            bivar_y = st.selectbox(
                "Variabila pentru axa Y",
                options=bivar_cols,
                index=bivar_cols.index("Duration") if "Duration" in bivar_cols else (1 if len(bivar_cols) > 1 else 0)
            )

        with col3:
            # Variabila pentru colorare
            bivar_color = st.selectbox(
                "Colorează după (opțional)",
                options=["Niciuna"] + ["Severity", "Weather_Condition", "Sunrise_Sunset", "State"],
                format_func=lambda x: {
                    "Niciuna": "Niciuna",
                    "Severity": "Severitate",
                    "Weather_Condition": "Condiție Meteo",
                    "Sunrise_Sunset": "Moment al Zilei",
                    "State": "Stat"
                }[x]
            )

        # Tipul de vizualizare
        viz_type = st.radio(
            "Tip de vizualizare",
            options=["Scatter Plot", "Hexbin", "Density Contour", "ECDF"]
        )

        # Limitează numărul de puncte pentru performanță
        sample_size = min(5000, len(filtered_df))
        if len(filtered_df) > sample_size:
            st.info(
                f"Pentru performanță, se afișează un eșantion aleatoriu de {sample_size} puncte din totalul de {len(filtered_df)}.")
            plot_df = filtered_df.sample(sample_size)
        else:
            plot_df = filtered_df

        # Creează vizualizarea selectată
        if viz_type == "Scatter Plot":
            if bivar_color != "Niciuna":
                fig = px.scatter(
                    plot_df,
                    x=bivar_x,
                    y=bivar_y,
                    color=bivar_color,
                    opacity=0.6,
                    title=f"Scatter Plot: {bivar_y} vs {bivar_x}",
                    labels={bivar_x: bivar_x, bivar_y: bivar_y}
                )
            else:
                fig = px.scatter(
                    plot_df,
                    x=bivar_x,
                    y=bivar_y,
                    opacity=0.6,
                    title=f"Scatter Plot: {bivar_y} vs {bivar_x}",
                    labels={bivar_x: bivar_x, bivar_y: bivar_y}
                )

        elif viz_type == "Hexbin":
            fig = px.density_heatmap(
                plot_df,
                x=bivar_x,
                y=bivar_y,
                nbinsx=30,
                nbinsy=30,
                marginal_x="histogram",
                marginal_y="histogram",
                title=f"Hexbin Plot: {bivar_y} vs {bivar_x}",
                labels={bivar_x: bivar_x, bivar_y: bivar_y}
            )

        elif viz_type == "Density Contour":
            fig = px.density_contour(
                plot_df,
                x=bivar_x,
                y=bivar_y,
                marginal_x="histogram",
                marginal_y="histogram",
                title=f"Density Contour: {bivar_y} vs {bivar_x}",
                labels={bivar_x: bivar_x, bivar_y: bivar_y}
            )

        elif viz_type == "ECDF":
            # Plot empirical cumulative distribution function
            fig = px.ecdf(
                plot_df,
                x=bivar_x,
                color=bivar_color if bivar_color != "Niciuna" else None,
                title=f"ECDF pentru {bivar_x}",
                labels={bivar_x: bivar_x}
            )

        st.plotly_chart(fig, use_container_width=True)

        # Calcuklează și afișează statistici despre relația dintre variabile
        if bivar_x != bivar_y and all(col in numeric_cols for col in [bivar_x, bivar_y]):
            st.subheader("Statistici despre Relația dintre Variabile")

            corr_pearson = filtered_df[[bivar_x, bivar_y]].corr().iloc[0, 1]
            corr_spearman = filtered_df[[bivar_x, bivar_y]].corr(method="spearman").iloc[0, 1]

            col1, col2 = st.columns(2)

            with col1:
                st.metric("Corelație Pearson", f"{corr_pearson:.4f}")

            with col2:
                st.metric("Corelație Spearman", f"{corr_spearman:.4f}")

            st.info("""
            **Interpretarea Coeficienților de Corelație:**
            - **Corelație Pearson** măsoară relația liniară între variabile. Valorile variază între -1 și 1.
              - 1: Corelație pozitivă perfectă
              - 0: Nicio corelație
              - -1: Corelație negativă perfectă

            - **Corelație Spearman** măsoară relația monotonă între variabile, fiind mai robustă la outlieri și 
              relații neliniare. Valorile variază tot între -1 și 1.
            """)

    elif analysis_type == "Analiza POI":
        st.subheader("Analiza Caracteristicilor de Infrastructură (POI)")

        st.markdown("""
        Această analiză explorează relația dintre accidente și caracteristicile de infrastructură din apropiere.
        POI (Point of Interest) se referă la elemente precum treceri de pietoni, semafoare, intersecții, etc.
        """)

        # Obține toate coloanele POI
        poi_columns = [
            'Amenity', 'Bump', 'Crossing', 'Give_Way', 'Junction',
            'No_Exit', 'Railway', 'Roundabout', 'Station', 'Stop',
            'Traffic_Calming', 'Traffic_Signal', 'Turning_Loop'
        ]

        # Verifică dacă avem aceste coloane în setul de date
        available_poi = [col for col in poi_columns if col in filtered_df.columns]

        if available_poi:
            # Calculează procentul de accidente în apropierea fiecărui POI
            poi_percentages = {}
            for col in available_poi:
                true_count = filtered_df[col].sum()
                total_count = len(filtered_df)
                percentage = (true_count / total_count) * 100
                poi_percentages[col] = percentage

            # Creează DataFrame pentru vizualizare
            poi_df = pd.DataFrame({
                'POI_Feature': list(poi_percentages.keys()),
                'Percentage': list(poi_percentages.values())
            }).sort_values('Percentage', ascending=False)

            # Vizualizează ca un grafic orizontal
            fig = px.bar(
                poi_df,
                y='POI_Feature',
                x='Percentage',
                orientation='h',
                title='Procentul de Accidente în Apropierea Caracteristicilor de Infrastructură',
                labels={'POI_Feature': 'Caracteristică de Infrastructură', 'Percentage': 'Procent (%)'},
                color='Percentage',
                color_continuous_scale='Blues'
            )

            st.plotly_chart(fig, use_container_width=True)

            # Analiza severității în funcție de POI
            st.subheader("Severitatea Accidentelor în Funcție de Caracteristicile de Infrastructură")

            # Selectează caracteristica pentru analiză
            selected_poi = st.selectbox(
                "Selectați caracteristica pentru analiză",
                options=available_poi,
                index=available_poi.index('Traffic_Signal') if 'Traffic_Signal' in available_poi else 0
            )

            # Creează un DataFrame pentru vizualizare
            poi_severity = filtered_df.groupby(selected_poi)['Severity'].mean().reset_index()
            poi_severity['POI_Status'] = poi_severity[selected_poi].map({True: "Prezent", False: "Absent"})

            fig = px.bar(
                poi_severity,
                x='POI_Status',
                y='Severity',
                color='Severity',
                title=f'Severitatea Medie a Accidentelor în Funcție de Prezența {selected_poi}',
                labels={'POI_Status': f'Prezența {selected_poi}', 'Severity': 'Severitate Medie'},
                color_continuous_scale='Reds'
            )

            st.plotly_chart(fig, use_container_width=True)

            # Analiză mai detaliată - distribuție BoxPlot
            st.subheader(f"Distribuția Severității pentru {selected_poi}")

            fig = px.box(
                filtered_df,
                x=selected_poi,
                y='Severity',
                color=selected_poi,
                points="all",
                title=f'Distribuția Severității în Funcție de Prezența {selected_poi}',
                labels={selected_poi: f'Prezența {selected_poi}', 'Severity': 'Severitate'}
            )

            st.plotly_chart(fig, use_container_width=True)

            # Analiza efectului combinat al POI
            st.subheader("Efectul Combinat al Caracteristicilor de Infrastructură")

            # Adaugă o coloană cu numărul de POI prezente
            filtered_df['POI_Count'] = filtered_df[available_poi].sum(axis=1)

            # Vizualizează relația dintre numărul de POI și severitate
            fig = px.box(
                filtered_df,
                x='POI_Count',
                y='Severity',
                color='POI_Count',
                title='Relația dintre Numărul de Caracteristici POI și Severitatea Accidentelor',
                labels={'POI_Count': 'Număr de Caracteristici POI', 'Severity': 'Severitate'}
            )

            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("Nu există coloane POI în setul de date filtrat.")

# Footer
st.markdown("---")
st.markdown("### 📊 Analiza Exploratorie a Datelor de Accidente Rutiere")
st.markdown(
    "Această aplicație demonstrează diverse tehnici de analiză exploratorie a datelor folosind Streamlit, Pandas, Plotly și alte biblioteci Python.")

# Export button
st.markdown("---")
st.header("📤 Opțiuni de Export")
if st.button("Generează Raport de Analiză"):
    with st.spinner("Se generează raportul PDF..."):
        # Aici ar trebui să fie cod pentru generarea PDF-ului
        st.success("Raport generat cu succes!")

st.download_button(
    label="Descarcă Datele Procesate",
    data=filtered_df.to_csv(index=False).encode('utf-8'),
    file_name='date_accidente_procesate.csv',
    mime='text/csv'
)