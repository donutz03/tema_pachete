import pandas as pd
import numpy as np

def debug_datetime_conversion():
    print("=== DEBUG: Testarea conversiei datetime pentru End_Time ===")
    
    # Încărcăm datele
    df = pd.read_csv('proiect/US_Accidents_Sample_1000_Per_Year.csv')
    
    # Extragem anul pentru filtrare
    df['Start_Time'] = pd.to_datetime(df['Start_Time'], errors='coerce')
    df['Year'] = df['Start_Time'].dt.year
    
    # Filtrăm pentru 2023
    df_2023 = df[df['Year'] == 2023].copy()
    print(f"Înregistrări 2023: {len(df_2023)}")
    
    # Testăm diferite metode de conversie
    print("\n1. Testare conversie cu errors='coerce' (metoda curentă)")
    df_2023['End_Time_coerce'] = pd.to_datetime(df_2023['End_Time'], errors='coerce')
    print(f"NaN după conversie cu 'coerce': {df_2023['End_Time_coerce'].isna().sum()}")
    
    print("\n2. Testare conversie cu errors='raise'")
    try:
        df_2023['End_Time_raise'] = pd.to_datetime(df_2023['End_Time'], errors='raise')
        print(f"NaN după conversie cu 'raise': {df_2023['End_Time_raise'].isna().sum()}")
    except Exception as e:
        print(f"Eroare la conversie cu 'raise': {e}")
    
    print("\n3. Testare conversie cu format explicit")
    # Analizăm formatul datelor
    sample_values = df_2023['End_Time'].head(5)
    print("Exemple de valori End_Time pentru 2023:")
    for val in sample_values:
        print(f"  '{val}' (lungime: {len(val)})")
    
    # Testăm cu format explicit
    try:
        df_2023['End_Time_format'] = pd.to_datetime(df_2023['End_Time'], 
                                                    format='%Y-%m-%d %H:%M:%S.%f')
        print(f"NaN după conversie cu format explicit: {df_2023['End_Time_format'].isna().sum()}")
    except Exception as e:
        print(f"Eroare la conversie cu format explicit: {e}")
    
    print("\n4. Testare prelucrare string înainte de conversie")
    # Eliminăm nanosecundele în exces
    df_2023['End_Time_cleaned'] = df_2023['End_Time'].str.replace('.000000000', '')
    print("Exemple după eliminarea nanosecundelor:")
    for val in df_2023['End_Time_cleaned'].head(5):
        print(f"  '{val}'")
    
    # Convertim valorile curățate
    df_2023['End_Time_cleaned_converted'] = pd.to_datetime(df_2023['End_Time_cleaned'], 
                                                           errors='coerce')
    print(f"NaN după curățare și conversie: {df_2023['End_Time_cleaned_converted'].isna().sum()}")
    
    print("\n5. Testare cu utc=True")
    try:
        df_2023['End_Time_utc'] = pd.to_datetime(df_2023['End_Time'], 
                                                 errors='coerce', utc=True)
        print(f"NaN după conversie cu utc=True: {df_2023['End_Time_utc'].isna().sum()}")
    except Exception as e:
        print(f"Eroare la conversie cu utc=True: {e}")
    
    print("\n6. Comparație cu anii anteriori")
    # Testăm cu un an anterior pentru comparație
    df_2022 = df[df['Year'] == 2022].copy()
    print(f"Înregistrări 2022: {len(df_2022)}")
    
    print("Exemple End_Time pentru 2022:")
    for val in df_2022['End_Time'].head(5):
        print(f"  '{val}' (lungime: {len(val)})")
    
    df_2022['End_Time_converted'] = pd.to_datetime(df_2022['End_Time'], errors='coerce')
    print(f"NaN în 2022 după conversie: {df_2022['End_Time_converted'].isna().sum()}")
    
    print("\n7. Analiză detaliată format")
    # Analizăm diferențele de format între ani
    unique_formats_2023 = df_2023['End_Time'].str.len().value_counts()
    unique_formats_2022 = df_2022['End_Time'].str.len().value_counts()
    
    print("Lungimi string End_Time în 2023:")
    print(unique_formats_2023)
    print("\nLungimi string End_Time în 2022:")
    print(unique_formats_2022)
    
    # Testăm conversie individuală pentru câteva valori
    print("\n8. Test conversie individuală")
    test_values_2023 = df_2023['End_Time'].head(3).tolist()
    test_values_2022 = df_2022['End_Time'].head(3).tolist()
    
    print("Testare valori 2023:")
    for val in test_values_2023:
        try:
            converted = pd.to_datetime(val)
            print(f"  '{val}' -> {converted} ✓")
        except Exception as e:
            print(f"  '{val}' -> ERROR: {e}")
    
    print("\nTestare valori 2022:")
    for val in test_values_2022:
        try:
            converted = pd.to_datetime(val)
            print(f"  '{val}' -> {converted} ✓")
        except Exception as e:
            print(f"  '{val}' -> ERROR: {e}")
    
    return df_2023, df_2022

if __name__ == "__main__":
    df_2023, df_2022 = debug_datetime_conversion() 