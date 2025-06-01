import pandas as pd

def test_after_csv_cleanup():
    """
    Testează că după curățarea CSV-ului, 
    aplicația funcționează fără fix-ul manual pentru End_Time
    """
    
    print("=== TEST: Verificarea după curățarea CSV ===")
    
    # Testăm exact funcția load_data FĂRĂ fix-ul manual
    df = pd.read_csv('proiect/US_Accidents_Sample_1000_Per_Year.csv')

    # Convertim coloanele de timp FĂRĂ fix-ul manual
    df['Start_Time'] = pd.to_datetime(df['Start_Time'], errors='coerce')
    df['End_Time'] = pd.to_datetime(df['End_Time'], errors='coerce')  # FĂRĂ curățare manuală

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
    
    print(f"📊 Înregistrări totale: {len(df)}")
    print(f"📊 Înregistrări filtrate pentru 2023: {len(filtered_df)}")
    
    # Verificăm valorile lipsă
    na_cols = filtered_df.columns[filtered_df.isna().any()].tolist()
    
    print(f"\n📋 Coloane cu valori lipsă: {len(na_cols)}")
    for col in na_cols:
        na_count = filtered_df[col].isna().sum()
        na_percent = (na_count / len(filtered_df) * 100)
        print(f"  {col}: {na_count} ({na_percent:.2f}%)")
    
    # Verificare specifică End_Time
    end_time_nan = filtered_df['End_Time'].isna().sum()
    print(f"\n🎯 End_Time NaN count: {end_time_nan}")
    
    if end_time_nan <= 1:  # Acceptăm 1 NaN maxim
        print("✅ SUCCESS: CSV-ul curățat funcționează perfect fără fix manual!")
        csv_clean = True
    else:
        print("❌ FAIL: Încă sunt probleme cu End_Time!")
        csv_clean = False
    
    # Verificăm Duration
    duration_nan = filtered_df['Duration'].isna().sum()
    print(f"🎯 Duration NaN count: {duration_nan}")
    
    # Afișăm câteva exemple pentru 2023
    print(f"\n🔍 Exemple de date din 2023 (după curățarea CSV):")
    sample = filtered_df[['Start_Time', 'End_Time', 'Duration']].head(3)
    for idx, row in sample.iterrows():
        print(f"  Start: {row['Start_Time']} | End: {row['End_Time']} | Duration: {row['Duration']:.1f} min")
    
    # Verificăm formatele actuale
    print(f"\n📏 Verificarea formatelor actuale în CSV:")
    
    # Reîncărcăm pentru a vedea formatele string
    df_raw = pd.read_csv('proiect/US_Accidents_Sample_1000_Per_Year.csv')
    end_time_lengths = df_raw['End_Time'].str.len().value_counts().sort_index()
    start_time_lengths = df_raw['Start_Time'].str.len().value_counts().sort_index()
    
    print("End_Time lungimi:")
    for length, count in end_time_lengths.items():
        print(f"  {length} caractere: {count} valori")
    
    print("Start_Time lungimi:")
    for length, count in start_time_lengths.items():
        print(f"  {length} caractere: {count} valori")
    
    return csv_clean, filtered_df

if __name__ == "__main__":
    is_clean, df_result = test_after_csv_cleanup()
    
    if is_clean:
        print("\n🎉 CONCLUZIE: CSV-ul a fost curățat cu succes!")
        print("💡 Acum poți elimina fix-ul manual din aplicația Streamlit")
        print("🔧 Aplicația va funcționa perfect cu formatul uniformizat")
    else:
        print("\n⚠️ ATENȚIE: Mai sunt probleme care necesită investigare") 