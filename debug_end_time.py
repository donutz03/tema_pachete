import pandas as pd
import numpy as np

def debug_end_time():
    print("=== DEBUG: Investigarea valorilor lipsă pentru End_Time ===")
    
    # Încărcăm datele brute
    print("\n1. Încărcarea datelor brute...")
    df = pd.read_csv('proiect/US_Accidents_Sample_1000_Per_Year.csv')
    print(f"Dimensiuni dataset original: {df.shape}")
    
    # Verificăm End_Time înainte de conversie
    print("\n2. Verificarea End_Time înainte de conversie la datetime...")
    print(f"Tipul coloanei End_Time: {df['End_Time'].dtype}")
    print(f"Valori null înainte de conversie: {df['End_Time'].isnull().sum()}")
    print(f"Valori NaN înainte de conversie: {df['End_Time'].isna().sum()}")
    print(f"Valori goale (string gol): {(df['End_Time'] == '').sum()}")
    print(f"Valori 'None' (string): {(df['End_Time'] == 'None').sum()}")
    print(f"Valori 'nan' (string): {(df['End_Time'] == 'nan').sum()}")
    print(f"Valori 'NaN' (string): {(df['End_Time'] == 'NaN').sum()}")
    
    # Vedem câteva exemple de valori End_Time
    print("\nPrimele 10 valori End_Time (brute):")
    print(df['End_Time'].head(10).tolist())
    
    print("\nUltimele 10 valori End_Time (brute):")
    print(df['End_Time'].tail(10).tolist())
    
    # Verificăm valorile unice pentru a vedea dacă există probleme
    unique_values = df['End_Time'].value_counts()
    print(f"\nNumărul de valori unice pentru End_Time: {len(unique_values)}")
    
    # Verificăm dacă există valori problematice
    problematic_values = df[df['End_Time'].astype(str).str.len() < 10]['End_Time'].value_counts()
    if len(problematic_values) > 0:
        print(f"\nValori End_Time cu lungime < 10 caractere: {len(problematic_values)}")
        print(problematic_values.head())
    
    # Convertim Start_Time pentru a extrage anul
    print("\n3. Convertirea Start_Time pentru a extrage anul...")
    df['Start_Time'] = pd.to_datetime(df['Start_Time'], errors='coerce')
    df['Year'] = df['Start_Time'].dt.year
    print(f"Distribuția pe ani:")
    print(df['Year'].value_counts().sort_index())
    
    # Filtrăm pentru 2023 înainte de conversie End_Time
    print("\n4. Filtrarea pentru anul 2023 (înainte de conversie End_Time)...")
    df_2023 = df[df['Year'] == 2023]
    print(f"Numărul de înregistrări pentru 2023: {len(df_2023)}")
    print(f"Valori null End_Time în 2023 (înainte de conversie): {df_2023['End_Time'].isnull().sum()}")
    print(f"Valori NaN End_Time în 2023 (înainte de conversie): {df_2023['End_Time'].isna().sum()}")
    
    # Verificăm valorile End_Time pentru 2023
    print("\nPrimele 10 valori End_Time pentru 2023 (înainte de conversie):")
    print(df_2023['End_Time'].head(10).tolist())
    
    # Acum convertim End_Time
    print("\n5. Convertirea End_Time la datetime...")
    df_original_end_time = df['End_Time'].copy()  # Salvăm originalul
    df['End_Time'] = pd.to_datetime(df['End_Time'], errors='coerce')
    
    # Verificăm câte valori au devenit NaN după conversie
    print(f"Valori NaN End_Time după conversie (tot dataset): {df['End_Time'].isna().sum()}")
    
    # Verificăm pentru 2023 după conversie
    df_2023_after = df[df['Year'] == 2023]
    print(f"Valori NaN End_Time în 2023 (după conversie): {df_2023_after['End_Time'].isna().sum()}")
    
    # Vedem care valori au devenit problematice
    print("\n6. Identificarea valorilor care au devenit NaN după conversie...")
    became_nan = df_original_end_time[df['End_Time'].isna() & ~pd.to_datetime(df_original_end_time, errors='coerce').isna()]
    
    if len(became_nan) > 0:
        print(f"Numărul de valori care au devenit NaN: {len(became_nan)}")
        print("Exemple de valori care au devenit NaN:")
        print(became_nan.value_counts().head(10))
    else:
        # Verificăm valorile care erau deja problematice
        original_problematic = df_original_end_time[pd.to_datetime(df_original_end_time, errors='coerce').isna()]
        print(f"Valori care erau deja problematice înainte de conversie: {len(original_problematic)}")
        if len(original_problematic) > 0:
            print("Exemple de valori problematice originale:")
            print(original_problematic.value_counts().head(10))
    
    # Verificăm specific pentru 2023
    print("\n7. Analiza detaliată pentru anul 2023...")
    df_2023_original = df[df['Year'] == 2023]
    original_end_times_2023 = df_original_end_time[df['Year'] == 2023]
    
    print(f"Total înregistrări 2023: {len(df_2023_original)}")
    print(f"End_Time NaN în 2023 după conversie: {df_2023_original['End_Time'].isna().sum()}")
    
    # Vedem valorile originale pentru End_Time în 2023 care au devenit NaN
    nan_indices_2023 = df_2023_original['End_Time'].isna()
    if nan_indices_2023.any():
        problematic_2023 = original_end_times_2023[nan_indices_2023]
        print(f"\nValori End_Time din 2023 care au devenit NaN:")
        print(problematic_2023.value_counts().head(20))
        
        # Verificăm câteva exemple specifice
        print(f"\nPrimele 10 valori problematice din 2023:")
        print(problematic_2023.head(10).tolist())
    
    # Verificăm dacă toate valorile din 2023 au aceeași problemă
    print(f"\n8. Verificare finală...")
    unique_end_times_2023 = original_end_times_2023.unique()
    print(f"Numărul de valori unice End_Time în 2023: {len(unique_end_times_2023)}")
    
    # Testăm conversia pentru câteva valori
    print(f"\nTestarea conversiei pentru primele 5 valori unice din 2023:")
    for i, val in enumerate(unique_end_times_2023[:5]):
        try:
            converted = pd.to_datetime(val, errors='raise')
            print(f"  {val} -> {converted} ✓")
        except Exception as e:
            print(f"  {val} -> ERROR: {e}")
    
    # Salvăm rezultatele pentru investigare ulterioară
    print(f"\n9. Salvarea rezultatelor pentru investigare...")
    
    # Creăm un dataset cu probleme pentru 2023
    problem_data_2023 = pd.DataFrame({
        'ID': df[df['Year'] == 2023]['ID'].values,
        'Original_End_Time': original_end_times_2023.values,
        'Converted_End_Time': df[df['Year'] == 2023]['End_Time'].values,
        'Is_NaN_After_Conversion': df[df['Year'] == 2023]['End_Time'].isna().values
    })
    
    problem_data_2023.to_csv('debug_end_time_2023.csv', index=False)
    print("Rezultatele au fost salvate în 'debug_end_time_2023.csv'")
    
    return df, problem_data_2023

if __name__ == "__main__":
    df, problem_data = debug_end_time() 