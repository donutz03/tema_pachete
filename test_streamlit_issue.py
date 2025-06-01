import pandas as pd
import streamlit as st

def test_streamlit_load():
    st.title("Test End_Time Issue")
    
    # Testez încărcarea datelor așa cum o face aplicația
    df = pd.read_csv('US_Accidents_Sample_1000_Per_Year.csv')
    
    # Convertim coloanele de timp exact ca în aplicație
    df['Start_Time'] = pd.to_datetime(df['Start_Time'], errors='coerce')
    df['End_Time'] = pd.to_datetime(df['End_Time'], errors='coerce')
    
    # Extragem anul pentru filtrare
    df['Year'] = df['Start_Time'].dt.year
    
    # Filtrăm pentru 2023
    filtered_df = df[df['Year'] == 2023]
    
    st.write(f"Total înregistrări 2023: {len(filtered_df)}")
    st.write(f"End_Time NaN în 2023: {filtered_df['End_Time'].isna().sum()}")
    
    # Afișăm informații despre End_Time
    st.subheader("Informații End_Time pentru 2023")
    
    # Afișăm primele valori
    st.write("Primele 10 valori End_Time:")
    st.write(filtered_df['End_Time'].head(10))
    
    # Verificăm tipul de date
    st.write(f"Tip date End_Time: {filtered_df['End_Time'].dtype}")
    
    # Afișăm statistici complete despre valori lipsă
    na_cols = filtered_df.columns[filtered_df.isna().any()].tolist()
    st.write("Coloane cu valori lipsă:")
    st.write(na_cols)
    
    if 'End_Time' in na_cols:
        st.error("End_Time este în lista coloanelor cu valori lipsă!")
    else:
        st.success("End_Time NU este în lista coloanelor cu valori lipsă!")

if __name__ == "__main__":
    test_streamlit_load() 