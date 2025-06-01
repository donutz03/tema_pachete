import pandas as pd
import sys
import os

# Adăugăm calea către directorul proiect
sys.path.append('proiect')

def test_fix():
    print("=== TEST: Verificarea fix-ului pentru End_Time ===")
    
    # Simulăm exact funcția load_data din aplicație
    df = pd.read_csv('proiect/US_Accidents_Sample_1000_Per_Year.csv')

    # Convertim coloanele de timp exact ca în aplicația fixată
    df['Start_Time'] = pd.to_datetime(df['Start_Time'], errors='coerce')
    
    # FIX: Curățăm valorile End_Time cu nanosecunde înainte de conversie
    df['End_Time'] = df['End_Time'].str.replace('.000000000', '', regex=False)
    df['End_Time'] = pd.to_datetime(df['End_Time'], errors='coerce')

    # Calculăm durata
    df['Duration'] = (df['End_Time'] - df['Start_Time']).dt.total_seconds() / 60

    # Convertim coloanele de tip object în string
    for col in df.select_dtypes(include='object').columns:
        if not pd.api.types.is_datetime64_any_dtype(df[col]):
            df[col] = df[col].astype(str)
    
    # Testăm filtrarea pentru 2023 (ca în aplicație)
    selected_years = (2023, 2023)
    severity_levels = [1, 2, 3, 4]

    filtered_df = df[
        (df['Start_Time'].dt.year >= selected_years[0]) &
        (df['Start_Time'].dt.year <= selected_years[1]) &
        (df['Severity'].isin(severity_levels))
    ]
    
    print(f"Înregistrări totale: {len(df)}")
    print(f"Înregistrări filtrate pentru 2023: {len(filtered_df)}")
    
    # Verificăm valorile lipsă
    na_cols = filtered_df.columns[filtered_df.isna().any()].tolist()
    
    print(f"\nColoane cu valori lipsă: {len(na_cols)}")
    for col in na_cols:
        na_count = filtered_df[col].isna().sum()
        na_percent = (na_count / len(filtered_df) * 100)
        print(f"  {col}: {na_count} ({na_percent:.2f}%)")
    
    # Verificare specifică End_Time
    end_time_nan = filtered_df['End_Time'].isna().sum()
    print(f"\n✅ End_Time NaN count: {end_time_nan} (Target: < 10)")
    
    if end_time_nan < 10:
        print("✅ SUCCESS: Fix-ul funcționează!")
    else:
        print("❌ FAIL: Fix-ul nu funcționează!")
    
    # Verificăm Duration
    duration_nan = filtered_df['Duration'].isna().sum()
    print(f"✅ Duration NaN count: {duration_nan}")
    
    # Afișăm câteva exemple
    print(f"\nExemple de date convertite cu succes:")
    sample = filtered_df[['Start_Time', 'End_Time', 'Duration']].head(3)
    for idx, row in sample.iterrows():
        print(f"  Start: {row['Start_Time']} | End: {row['End_Time']} | Duration: {row['Duration']:.1f} min")
    
    return filtered_df

if __name__ == "__main__":
    df_test = test_fix() 