import pandas as pd
import numpy as np

def debug_streamlit_exact():
    print("=== DEBUG: Reproducerea exactă a pașilor din aplicația Streamlit ===")
    
    # Pasul 1: Încărcarea datelor exact ca în aplicație
    print("\n1. Încărcarea datelor (ca în aplicație)...")
    df = pd.read_csv('proiect/US_Accidents_Sample_1000_Per_Year.csv')
    print(f"Dimensiuni dataset: {df.shape}")
    
    # Pasul 2: Convertirea coloanelor de timp exact ca în aplicație
    print("\n2. Convertirea coloanelor de timp...")
    print(f"End_Time înainte de conversie - tip: {df['End_Time'].dtype}")
    print(f"End_Time înainte - NaN count: {df['End_Time'].isna().sum()}")
    
    # Convertim coloanele de timp
    df['Start_Time'] = pd.to_datetime(df['Start_Time'], errors='coerce')
    df['End_Time'] = pd.to_datetime(df['End_Time'], errors='coerce')
    
    print(f"End_Time după conversie - tip: {df['End_Time'].dtype}")
    print(f"End_Time după conversie - NaN count: {df['End_Time'].isna().sum()}")
    
    # Pasul 3: Calcularea duratei exact ca în aplicație
    print("\n3. Calcularea duratei...")
    df['Duration'] = (df['End_Time'] - df['Start_Time']).dt.total_seconds() / 60
    print(f"Duration calculată - NaN count: {df['Duration'].isna().sum()}")
    
    # Pasul 4: Convertirea coloanelor object în string exact ca în aplicație
    print("\n4. Convertirea coloanelor object în string...")
    for col in df.select_dtypes(include='object').columns:
        if not pd.api.types.is_datetime64_any_dtype(df[col]):
            print(f"  Convertind {col} la string...")
            df[col] = df[col].astype(str)
    
    print(f"End_Time după conversiile object - NaN count: {df['End_Time'].isna().sum()}")
    
    # Pasul 5: Filtrarea pentru anii selectați exact ca în aplicație
    print("\n5. Filtrarea pentru anii selectați...")
    selected_years = (2023, 2023)  # Valoarea default din aplicație
    severity_levels = [1, 2, 3, 4]  # Valoarea default din aplicație
    
    print(f"Start_Time tip după conversie: {df['Start_Time'].dtype}")
    print(f"Start_Time NaN după conversie: {df['Start_Time'].isna().sum()}")
    
    # Filtrarea exact ca în aplicație
    filtered_df = df[
        (df['Start_Time'].dt.year >= selected_years[0]) &
        (df['Start_Time'].dt.year <= selected_years[1]) &
        (df['Severity'].isin(severity_levels))
    ]
    
    print(f"Dimensiuni după filtrare: {filtered_df.shape}")
    
    # Pasul 6: Verificarea valorilor lipsă exact ca în aplicație
    print("\n6. Verificarea valorilor lipsă...")
    na_cols = filtered_df.columns[filtered_df.isna().any()].tolist()
    
    print(f"Coloane cu valori lipsă: {len(na_cols)}")
    for col in na_cols:
        na_count = filtered_df[col].isna().sum()
        na_percent = (na_count / len(filtered_df) * 100)
        print(f"  {col}: {na_count} ({na_percent:.2f}%)")
    
    # Verificare specifică pentru End_Time
    print(f"\n7. Verificare detaliată End_Time...")
    print(f"End_Time în filtered_df - tip: {filtered_df['End_Time'].dtype}")
    print(f"End_Time în filtered_df - NaN count: {filtered_df['End_Time'].isna().sum()}")
    print(f"End_Time în filtered_df - Total count: {len(filtered_df)}")
    
    # Verificăm câteva valori End_Time din filtered_df
    print(f"\nPrimele 5 valori End_Time din filtered_df:")
    for i, val in enumerate(filtered_df['End_Time'].head(5)):
        print(f"  {i}: {val} (tip: {type(val)}) (is NaN: {pd.isna(val)})")
    
    # Verificăm dacă toate valorile sunt NaN
    all_nan = filtered_df['End_Time'].isna().all()
    print(f"\nToate valorile End_Time sunt NaN: {all_nan}")
    
    # Verificăm valoarea exactă a unei înregistrări
    if len(filtered_df) > 0:
        sample_row = filtered_df.iloc[0]
        print(f"\nExemplu de înregistrare (prima):")
        print(f"  ID: {sample_row['ID']}")
        print(f"  Start_Time: {sample_row['Start_Time']} (tip: {type(sample_row['Start_Time'])})")
        print(f"  End_Time: {sample_row['End_Time']} (tip: {type(sample_row['End_Time'])})")
        print(f"  Duration: {sample_row['Duration']}")
    
    return df, filtered_df

if __name__ == "__main__":
    df_original, df_filtered = debug_streamlit_exact() 