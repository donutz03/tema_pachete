from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import numpy as np
import streamlit as st
import seaborn as sns

def aplica_scalare(df, numeric_cols, metoda='standard'):
    """
    Funcție pentru aplicarea diferitelor metode de scalare pe coloanele numerice ale unui DataFrame.

    Parametri:
    -----------
    df : pd.DataFrame
        DataFrame-ul care conține datele
    numeric_cols : list
        Lista coloanelor numerice care vor fi scalate
    metoda : str
        Metoda de scalare: 'standard', 'minmax', 'robust', 'log'

    Return:
    -----------
    df_scaled : pd.DataFrame
        DataFrame-ul cu datele scalate
    """
    df_scaled = df.copy()

    if metoda == 'standard':
        # Standardizare (Z-score): medie 0, deviație standard 1
        scaler = StandardScaler()
        df_scaled[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        descriere = "Standardizare (Z-score): transformă datele pentru a avea medie 0 și deviație standard 1"

    elif metoda == 'minmax':
        # MinMax scaling: scalează datele în intervalul [0,1]
        scaler = MinMaxScaler()
        df_scaled[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        descriere = "Normalizare MinMax: scalează datele în intervalul [0,1]"

    elif metoda == 'robust':
        # Robust scaling: utilizează mediana și IQR, mai puțin sensibil la outlieri
        scaler = RobustScaler()
        df_scaled[numeric_cols] = scaler.fit_transform(df[numeric_cols])
        descriere = "Scalare robustă: utilizează mediana și IQR în loc de medie și deviație standard"

    elif metoda == 'log':
        # Transformare logaritmică: pentru date cu distribuție asimetrică pozitivă
        # Adăugăm 1 pentru a gestiona valorile 0 (log(0) este nedefinit)
        for col in numeric_cols:
            # Verificăm dacă există valori negative sau zero
            min_val = df[col].min()
            if min_val <= 0:
                # Adăugăm o constantă pentru a face toate valorile pozitive
                constant = abs(min_val) + 1 if min_val < 0 else 1
                df_scaled[col] = np.log(df[col] + constant)
            else:
                df_scaled[col] = np.log(df[col])
        descriere = "Transformare logaritmică: reduce asimetria și efectul outlierilor pentru distribuții asimetrice"

    return df_scaled, descriere


def adauga_sectiune_scalare(df, sidebar=False):
    """
    Adaugă o secțiune pentru scalarea datelor în aplicația Streamlit

    Parametri:
    -----------
    df : pd.DataFrame
        DataFrame-ul care conține datele
    sidebar : bool
        Dacă True, adaugă secțiunea în sidebar, altfel în main area
    """
    container = st.sidebar if sidebar else st

    # Selectăm doar coloanele numerice pentru scalare
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # Selectăm coloana pentru scalare
    col_for_scaling = st.selectbox("Selectează coloana pentru scalare", numeric_cols)

    # Afișăm statisticile și distribuția coloanei originale
    st.subheader(f"Distribuția originală pentru {col_for_scaling}")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.markdown("### Statistici")
        st.dataframe(df[col_for_scaling].describe(), use_container_width=True)

    with col2:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(df[col_for_scaling], kde=True, ax=ax)
        ax.set_title(f"Distribuția originală: {col_for_scaling}")
        st.pyplot(fig)

    # Selectăm metoda de scalare
    st.subheader("Alege metoda de scalare")

    scaling_method = st.radio(
        "Metodă de scalare:",
        ["standard", "minmax", "robust", "log"]
    )

    if st.button("Aplică Scalarea"):
        # Aplicăm scalarea
        df_scaled, descriere = aplica_scalare(df, [col_for_scaling], scaling_method)

        # Afișăm descrierea metodei
        st.info(descriere)

        # Afișăm statisticile după scalare
        st.subheader(f"Distribuția după scalare ({scaling_method})")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("##### Statistici după scalare")
            st.dataframe(df_scaled[col_for_scaling].describe(), use_container_width=True)

            # Adaugă metrici specifice metodei de scalare
            if scaling_method == "standard":
                st.metric("Media după standardizare", f"{df_scaled[col_for_scaling].mean():.6f}")
                st.metric("Deviația standard după standardizare", f"{df_scaled[col_for_scaling].std():.6f}")
            elif scaling_method == "minmax":
                st.metric("Valoare minimă după normalizare", f"{df_scaled[col_for_scaling].min():.6f}")
                st.metric("Valoare maximă după normalizare", f"{df_scaled[col_for_scaling].max():.6f}")

        with col2:
            # Vizualizăm distribuția după scalare
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(df_scaled[col_for_scaling], kde=True, ax=ax)
            ax.set_title(f"Distribuția după {scaling_method}: {col_for_scaling}")
            st.pyplot(fig)

        # Adaugă grafic comparativ
        st.subheader("Comparație înainte și după scalare")
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))

        # Grafic original
        sns.histplot(df[col_for_scaling], kde=True, ax=axes[0])
        axes[0].set_title(f"Original: {col_for_scaling}")

        # Grafic după scalare
        sns.histplot(df_scaled[col_for_scaling], kde=True, ax=axes[1])
        axes[1].set_title(f"După {scaling_method}")

        plt.tight_layout()
        st.pyplot(fig)

        # Opțiune pentru descărcare
        st.download_button(
            label="Descarcă datele scalate (CSV)",
            data=df_scaled.to_csv(index=False),
            file_name=f"date_scalate_{scaling_method}.csv",
            mime="text/csv"
        )