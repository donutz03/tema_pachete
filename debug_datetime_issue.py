import pandas as pd
import numpy as np

def debug_datetime_conversion_issue():
    print("=== DEBUG: Problema specifică cu conversia datetime ===")
    
    # Încărcăm datele
    df = pd.read_csv('proiect/US_Accidents_Sample_1000_Per_Year.csv')
    
    # Filtrăm pentru 2023 înainte de conversie
    df['Start_Time'] = pd.to_datetime(df['Start_Time'], errors='coerce')
    df['Year'] = df['Start_Time'].dt.year
    df_2023 = df[df['Year'] == 2023]
    
    print(f"Înregistrări pentru 2023: {len(df_2023)}")
    
    # Analizăm valorile End_Time înainte de conversie
    print("\n1. Analiza valorilor End_Time din 2023 înainte de conversie...")
    sample_end_times = df_2023['End_Time'].head(10)
    
    for i, val in enumerate(sample_end_times):
        print(f"  {i}: '{val}' (lungime: {len(val)})")
    
    # Testăm conversia individuală
    print("\n2. Testarea conversiei individuale...")
    
    test_values = [
        '2023-03-31 18:09:49.000000000',  # Format cu nanosecunde
        '2023-03-31 18:09:49',           # Format fără nanosecunde
        '2023-03-31 18:09:49.000000',    # Format cu microsecunde
        '2023-03-31 18:09:49.000',       # Format cu milisecunde
    ]
    
    for val in test_values:
        try:
            converted = pd.to_datetime(val, errors='raise')
            print(f"  '{val}' -> {converted} ✓")
        except Exception as e:
            print(f"  '{val}' -> ERROR: {e}")
    
    # Testăm conversia cu DataFrame
    print("\n3. Testarea conversiei cu DataFrame...")
    
    test_df = pd.DataFrame({
        'test_datetime': [
            '2023-03-31 18:09:49.000000000',
            '2023-03-31 18:09:50.000000000',
            '2023-03-31 18:09:51.000000000'
        ]
    })
    
    print("Înainte de conversie:")
    print(test_df['test_datetime'])
    
    test_df['converted'] = pd.to_datetime(test_df['test_datetime'], errors='coerce')
    
    print("\nDupă conversie:")
    print(test_df['converted'])
    print(f"NaN count: {test_df['converted'].isna().sum()}")
    
    # Testăm cu seria originală din 2023
    print("\n4. Testarea cu seria originală din 2023...")
    
    # Luăm primele 5 valori End_Time din 2023
    original_series = df_2023['End_Time'].head(5).copy()
    print("Seria originală:")
    print(original_series)
    
    # Convertim seria
    converted_series = pd.to_datetime(original_series, errors='coerce')
    print("\nSeria convertită:")
    print(converted_series)
    print(f"NaN count: {converted_series.isna().sum()}")
    
    # Testăm cu diferite metode de conversie
    print("\n5. Testarea cu diferite metode de conversie...")
    
    # Metoda 1: errors='coerce'
    method1 = pd.to_datetime(original_series, errors='coerce')
    print(f"Metoda 1 (errors='coerce') - NaN: {method1.isna().sum()}")
    
    # Metoda 2: cu format explicit
    try:
        method2 = pd.to_datetime(original_series, format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')
        print(f"Metoda 2 (format explicit) - NaN: {method2.isna().sum()}")
    except Exception as e:
        print(f"Metoda 2 - ERROR: {e}")
    
    # Metoda 3: curățare string înainte de conversie
    cleaned_series = original_series.str.replace('.000000000', '', regex=False)
    method3 = pd.to_datetime(cleaned_series, errors='coerce')
    print(f"Metoda 3 (după curățare) - NaN: {method3.isna().sum()}")
    
    # Metoda 4: înlocuire parțială
    replaced_series = original_series.str.replace('.000000000', '.000000', regex=False)
    method4 = pd.to_datetime(replaced_series, errors='coerce')
    print(f"Metoda 4 (înlocuire cu microsecunde) - NaN: {method4.isna().sum()}")
    
    # Verificăm versiunea pandas
    print(f"\n6. Informații despre mediu...")
    print(f"Versiunea pandas: {pd.__version__}")
    print(f"Versiunea numpy: {np.__version__}")
    
    # Testăm cu un exemplu specific
    print(f"\n7. Test specific cu o valoare...")
    test_value = '2023-03-31 18:09:49.000000000'
    
    # Test individual
    try:
        individual_result = pd.to_datetime(test_value)
        print(f"Test individual: {test_value} -> {individual_result}")
    except Exception as e:
        print(f"Test individual ERROR: {e}")
    
    # Test cu Series
    test_series = pd.Series([test_value])
    series_result = pd.to_datetime(test_series, errors='coerce')
    print(f"Test cu Series: NaN count = {series_result.isna().sum()}")
    
    return df_2023, original_series, method1, method2, method3, method4

if __name__ == "__main__":
    results = debug_datetime_conversion_issue() 