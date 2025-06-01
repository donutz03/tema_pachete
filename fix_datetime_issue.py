import pandas as pd
import numpy as np

def fix_end_time_conversion():
    print("=== SOLUȚIE: Fixarea conversiei End_Time ===")
    
    # Încărcăm datele
    df = pd.read_csv('proiect/US_Accidents_Sample_1000_Per_Year.csv')
    print(f"Dimensiuni dataset: {df.shape}")
    
    # Convertim Start_Time
    df['Start_Time'] = pd.to_datetime(df['Start_Time'], errors='coerce')
    
    # === SOLUȚIA 1: Curățarea valorilor End_Time înainte de conversie ===
    print("\n=== SOLUȚIA 1: Curățarea valorilor End_Time ===")
    
    # Verificăm formatele End_Time
    print("Verificarea formatelor End_Time...")
    lengths = df['End_Time'].str.len().value_counts().sort_index()
    print("Lungimi string End_Time:")
    print(lengths)
    
    # Curățăm valorile cu nanosecunde (29 caractere)
    print("\nCurățarea valorilor cu nanosecunde...")
    df['End_Time_cleaned'] = df['End_Time'].str.replace('.000000000', '', regex=False)
    
    # Verificăm noile lungimi
    new_lengths = df['End_Time_cleaned'].str.len().value_counts().sort_index()
    print("Lungimi după curățare:")
    print(new_lengths)
    
    # Convertim valorile curățate
    df['End_Time_fixed'] = pd.to_datetime(df['End_Time_cleaned'], errors='coerce')
    
    # Verificăm rezultatul
    df['Year'] = df['Start_Time'].dt.year
    
    print("\nRezultate conversie după curățare:")
    for year in [2022, 2023]:
        year_data = df[df['Year'] == year]
        nan_count = year_data['End_Time_fixed'].isna().sum()
        total_count = len(year_data)
        print(f"   {year}: {nan_count} NaN din {total_count} total ({nan_count/total_count*100:.1f}%)")
    
    # === SOLUȚIA 2: Conversie separată pe formate ===
    print("\n=== SOLUȚIA 2: Conversie separată pe formate ===")
    
    df_solution2 = pd.read_csv('proiect/US_Accidents_Sample_1000_Per_Year.csv')
    df_solution2['Start_Time'] = pd.to_datetime(df_solution2['Start_Time'], errors='coerce')
    df_solution2['Year'] = df_solution2['Start_Time'].dt.year
    
    # Separăm pe formate
    mask_long = df_solution2['End_Time'].str.len() == 29  # Format cu nanosecunde
    mask_short = df_solution2['End_Time'].str.len() == 19  # Format standard
    
    print(f"Valori cu format lung (29 char): {mask_long.sum()}")
    print(f"Valori cu format scurt (19 char): {mask_short.sum()}")
    
    # Inițializăm coloana End_Time cu NaT
    df_solution2['End_Time_fixed'] = pd.NaT
    
    # Convertim valorile cu format scurt (standard)
    df_solution2.loc[mask_short, 'End_Time_fixed'] = pd.to_datetime(
        df_solution2.loc[mask_short, 'End_Time'], errors='coerce'
    )
    
    # Convertim valorile cu format lung (curățate)
    df_solution2.loc[mask_long, 'End_Time_fixed'] = pd.to_datetime(
        df_solution2.loc[mask_long, 'End_Time'].str.replace('.000000000', '', regex=False), 
        errors='coerce'
    )
    
    print("\nRezultate conversie separată:")
    for year in [2022, 2023]:
        year_data = df_solution2[df_solution2['Year'] == year]
        nan_count = year_data['End_Time_fixed'].isna().sum()
        total_count = len(year_data)
        print(f"   {year}: {nan_count} NaN din {total_count} total ({nan_count/total_count*100:.1f}%)")
    
    # === SOLUȚIA 3: Funcție robustă de conversie ===
    print("\n=== SOLUȚIA 3: Funcție robustă de conversie ===")
    
    def convert_end_time_robust(series):
        """Convertește End_Time ținând cont de formatele mixte"""
        result = pd.Series(index=series.index, dtype='datetime64[ns]')
        
        # Încercăm conversia directă
        try:
            result = pd.to_datetime(series, errors='coerce')
            nan_count = result.isna().sum()
            
            # Dacă avem prea multe NaN, încercăm curățarea
            if nan_count > len(series) * 0.1:  # Peste 10% NaN
                print(f"   Detectate {nan_count} NaN, aplicăm curățarea...")
                cleaned_series = series.str.replace('.000000000', '', regex=False)
                result = pd.to_datetime(cleaned_series, errors='coerce')
                
        except Exception as e:
            print(f"   Eroare la conversie: {e}")
            result = pd.Series([pd.NaT] * len(series), index=series.index)
        
        return result
    
    df_solution3 = pd.read_csv('proiect/US_Accidents_Sample_1000_Per_Year.csv')
    df_solution3['Start_Time'] = pd.to_datetime(df_solution3['Start_Time'], errors='coerce')
    df_solution3['Year'] = df_solution3['Start_Time'].dt.year
    
    print("Aplicarea funcției robuste...")
    df_solution3['End_Time_fixed'] = convert_end_time_robust(df_solution3['End_Time'])
    
    print("\nRezultate conversie robustă:")
    for year in [2022, 2023]:
        year_data = df_solution3[df_solution3['Year'] == year]
        nan_count = year_data['End_Time_fixed'].isna().sum()
        total_count = len(year_data)
        print(f"   {year}: {nan_count} NaN din {total_count} total ({nan_count/total_count*100:.1f}%)")
    
    # Calculăm durata cu noile valori
    df_solution3['Duration_fixed'] = (df_solution3['End_Time_fixed'] - df_solution3['Start_Time']).dt.total_seconds() / 60
    
    print(f"\nDuration NaN count: {df_solution3['Duration_fixed'].isna().sum()}")
    
    # Afișăm câteva exemple
    print("\nExemple de conversie reușită:")
    sample_2023 = df_solution3[df_solution3['Year'] == 2023].head(3)
    for idx, row in sample_2023.iterrows():
        print(f"   Start: {row['Start_Time']}")
        print(f"   End:   {row['End_Time_fixed']}")
        print(f"   Durata: {row['Duration_fixed']:.1f} minute")
        print()
    
    return df, df_solution2, df_solution3

if __name__ == "__main__":
    df1, df2, df3 = fix_end_time_conversion() 