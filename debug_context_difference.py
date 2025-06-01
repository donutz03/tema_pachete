import pandas as pd
import numpy as np

def debug_context_difference():
    print("=== DEBUG: Compararea contextului de lucru ===")
    
    # Încărcăm datele
    df = pd.read_csv('proiect/US_Accidents_Sample_1000_Per_Year.csv')
    print(f"Dimensiuni dataset: {df.shape}")
    
    # === CONTEXT 1: Test simplu (care funcționează) ===
    print("\n=== CONTEXT 1: Test simplu (care funcționează) ===")
    
    # Luăm doar primele câteva valori End_Time pentru 2023
    df_temp = df.copy()
    df_temp['Start_Time'] = pd.to_datetime(df_temp['Start_Time'], errors='coerce')
    df_temp['Year'] = df_temp['Start_Time'].dt.year
    df_2023_simple = df_temp[df_temp['Year'] == 2023]
    
    end_time_sample = df_2023_simple['End_Time'].head(5)
    print("Seria End_Time (context simplu):")
    print(end_time_sample)
    
    converted_simple = pd.to_datetime(end_time_sample, errors='coerce')
    print(f"\nDupă conversie - NaN count: {converted_simple.isna().sum()}")
    print("Valorile convertite:")
    print(converted_simple)
    
    # === CONTEXT 2: Context complex (ca în aplicația Streamlit) ===
    print("\n=== CONTEXT 2: Context complex (ca în aplicația Streamlit) ===")
    
    # Reproducem exact pașii din aplicație
    df_complex = df.copy()
    
    # Pasul 1: Convertim Start_Time
    print("1. Convertire Start_Time...")
    df_complex['Start_Time'] = pd.to_datetime(df_complex['Start_Time'], errors='coerce')
    
    # Pasul 2: Convertim End_Time (aici apare problema)
    print("2. Convertire End_Time...")
    print(f"   End_Time înainte - tip: {df_complex['End_Time'].dtype}")
    print(f"   End_Time înainte - sample:")
    for i, val in enumerate(df_complex['End_Time'].head(3)):
        print(f"     {i}: '{val}' (tip: {type(val)})")
    
    # Verificăm dacă există diferențe în seria End_Time
    df_complex['Year'] = df_complex['Start_Time'].dt.year
    df_2023_complex = df_complex[df_complex['Year'] == 2023]
    
    print(f"   Numărul de înregistrări 2023: {len(df_2023_complex)}")
    
    # Verificăm seria End_Time din 2023 înainte de conversie
    end_time_2023_before = df_2023_complex['End_Time']
    print(f"   End_Time 2023 înainte - tip: {end_time_2023_before.dtype}")
    print(f"   End_Time 2023 înainte - primele 3 valori:")
    for i, val in enumerate(end_time_2023_before.head(3)):
        print(f"     {i}: '{val}' (tip: {type(val)}, lungime: {len(val)})")
    
    # ACUM facem conversia End_Time pentru tot dataset-ul
    print("   Efectuând conversia End_Time pentru tot dataset-ul...")
    df_complex['End_Time'] = pd.to_datetime(df_complex['End_Time'], errors='coerce')
    
    # Verificăm rezultatul pentru 2023
    df_2023_after = df_complex[df_complex['Year'] == 2023]
    end_time_2023_after = df_2023_after['End_Time']
    print(f"   End_Time 2023 după conversie - NaN count: {end_time_2023_after.isna().sum()}")
    print(f"   End_Time 2023 după conversie - total: {len(end_time_2023_after)}")
    
    # === INVESTIGAȚIE SUPLIMENTARĂ ===
    print("\n=== INVESTIGAȚIE SUPLIMENTARĂ ===")
    
    # Verificăm dacă există diferențe în valorile din 2023 vs alte ani
    print("1. Compararea formatelor End_Time pe ani...")
    
    for year in [2022, 2023]:
        year_data = df[df_temp['Year'] == year]
        if len(year_data) > 0:
            sample_val = year_data['End_Time'].iloc[0]
            print(f"   {year}: '{sample_val}' (lungime: {len(sample_val)})")
    
    # Verificăm dacă există caractere invizibile
    print("\n2. Verificarea caracterelor invizibile în End_Time 2023...")
    
    sample_2023_val = df_2023_simple['End_Time'].iloc[0]
    print(f"   Valoarea: '{sample_2023_val}'")
    print(f"   Coduri ASCII: {[ord(c) for c in sample_2023_val]}")
    
    # Verificăm dacă este o problemă cu index-ul
    print("\n3. Verificarea problemei cu index-ul...")
    
    # Reset index pentru End_Time din 2023
    end_time_reset = df_2023_simple['End_Time'].reset_index(drop=True)
    converted_reset = pd.to_datetime(end_time_reset, errors='coerce')
    print(f"   După reset index - NaN count: {converted_reset.isna().sum()}")
    
    # Testăm cu o serie nouă construită manual
    manual_series = pd.Series(df_2023_simple['End_Time'].tolist())
    converted_manual = pd.to_datetime(manual_series, errors='coerce')
    print(f"   Serie manuală - NaN count: {converted_manual.isna().sum()}")
    
    # === TESTĂM CU TOT DATASET-UL ===
    print("\n=== TESTĂM CU TOT DATASET-UL ===")
    
    df_full_test = df.copy()
    
    # Convertim Start_Time pentru a obține Year
    df_full_test['Start_Time'] = pd.to_datetime(df_full_test['Start_Time'], errors='coerce')
    
    # Convertim End_Time pentru tot dataset-ul
    print("Convertire End_Time pentru tot dataset-ul...")
    df_full_test['End_Time'] = pd.to_datetime(df_full_test['End_Time'], errors='coerce')
    
    # Adăugăm coloana Year după conversie
    df_full_test['Year'] = df_full_test['Start_Time'].dt.year
    
    # Verificăm rezultatul pentru fiecare an
    for year in [2022, 2023]:
        year_data = df_full_test[df_full_test['Year'] == year]
        nan_count = year_data['End_Time'].isna().sum()
        total_count = len(year_data)
        print(f"   {year}: {nan_count} NaN din {total_count} total ({nan_count/total_count*100:.1f}%)")
    
    return df_complex, df_full_test

if __name__ == "__main__":
    df_complex, df_full = debug_context_difference() 