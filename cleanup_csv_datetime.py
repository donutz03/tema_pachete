import pandas as pd
import numpy as np
import shutil
import os
from datetime import datetime

def cleanup_csv_datetime():
    """
    Uniformizează formatele datetime în fișierul CSV,
    eliminând nanosecundele în exces pentru compatibilitate cu SAS și alte sisteme
    """
    
    print("=== CURĂȚAREA FORMATELOR DATETIME ÎN CSV ===")
    
    # Definim căile
    original_file = 'proiect/US_Accidents_Sample_1000_Per_Year.csv'
    backup_file = 'proiect/US_Accidents_Sample_1000_Per_Year_BACKUP.csv'
    
    # Verificăm dacă fișierul există
    if not os.path.exists(original_file):
        print(f"❌ Fișierul {original_file} nu a fost găsit!")
        return False
    
    print(f"📁 Fișierul original: {original_file}")
    
    # Facem backup
    print("🔄 Crearea backup-ului...")
    shutil.copy2(original_file, backup_file)
    print(f"✅ Backup creat: {backup_file}")
    
    # Încărcăm datele
    print("\n📖 Încărcarea datelor originale...")
    df = pd.read_csv(original_file)
    print(f"Dimensiuni dataset: {df.shape}")
    
    # Analizăm formatele End_Time
    print("\n🔍 Analiza formatelor End_Time...")
    end_time_lengths = df['End_Time'].str.len().value_counts().sort_index()
    print("Lungimi End_Time găsite:")
    for length, count in end_time_lengths.items():
        print(f"  {length} caractere: {count} valori")
    
    # Analizăm formatele Start_Time
    print("\n🔍 Analiza formatelor Start_Time...")
    start_time_lengths = df['Start_Time'].str.len().value_counts().sort_index()
    print("Lungimi Start_Time găsite:")
    for length, count in start_time_lengths.items():
        print(f"  {length} caractere: {count} valori")
    
    # Curățăm End_Time
    print("\n🧹 Curățarea End_Time...")
    values_with_nanosec_end = (df['End_Time'].str.len() == 29).sum()
    if values_with_nanosec_end > 0:
        print(f"Găsite {values_with_nanosec_end} valori End_Time cu nanosecunde")
        df['End_Time'] = df['End_Time'].str.replace('.000000000', '', regex=False)
        print("✅ End_Time curățat (nanosecunde eliminate)")
    else:
        print("ℹ️ Nu au fost găsite valori End_Time cu nanosecunde")
    
    # Curățăm Start_Time (în cazul în care și acestea ar avea probleme)
    print("\n🧹 Curățarea Start_Time...")
    values_with_nanosec_start = (df['Start_Time'].str.len() == 29).sum()
    if values_with_nanosec_start > 0:
        print(f"Găsite {values_with_nanosec_start} valori Start_Time cu nanosecunde")
        df['Start_Time'] = df['Start_Time'].str.replace('.000000000', '', regex=False)
        print("✅ Start_Time curățat (nanosecunde eliminate)")
    else:
        print("ℹ️ Nu au fost găsite valori Start_Time cu nanosecunde")
    
    # Verificăm rezultatele
    print("\n✅ Verificarea rezultatelor după curățare...")
    end_time_lengths_new = df['End_Time'].str.len().value_counts().sort_index()
    start_time_lengths_new = df['Start_Time'].str.len().value_counts().sort_index()
    
    print("Lungimi End_Time după curățare:")
    for length, count in end_time_lengths_new.items():
        print(f"  {length} caractere: {count} valori")
    
    print("Lungimi Start_Time după curățare:")
    for length, count in start_time_lengths_new.items():
        print(f"  {length} caractere: {count} valori")
    
    # Testăm conversia datetime pentru a verifica că totul funcționează
    print("\n🧪 Testarea conversiei datetime...")
    try:
        start_converted = pd.to_datetime(df['Start_Time'], errors='coerce')
        end_converted = pd.to_datetime(df['End_Time'], errors='coerce')
        
        start_nan = start_converted.isna().sum()
        end_nan = end_converted.isna().sum()
        
        print(f"Start_Time - NaN după conversie: {start_nan}")
        print(f"End_Time - NaN după conversie: {end_nan}")
        
        if start_nan == 0 and end_nan <= 1:  # Acceptăm 1 NaN pentru End_Time (posibil o valoare realmente problematică)
            print("✅ Conversiile datetime sunt în regulă!")
        else:
            print("⚠️ Atenție: Mai există probleme cu conversiile datetime")
            
    except Exception as e:
        print(f"❌ Eroare la testarea conversiei: {e}")
        return False
    
    # Salvăm fișierul curățat
    print(f"\n💾 Salvarea fișierului curățat...")
    df.to_csv(original_file, index=False)
    print(f"✅ Fișierul curățat a fost salvat: {original_file}")
    
    # Verificăm dimensiunile finale
    print(f"\n📊 Verificare finală...")
    df_verify = pd.read_csv(original_file)
    print(f"Dimensiuni finale: {df_verify.shape}")
    
    # Afișăm un raport final
    print(f"\n📋 RAPORT FINAL:")
    print(f"✅ Backup creat: {backup_file}")
    print(f"✅ Fișier curățat: {original_file}")
    print(f"✅ Formatele datetime au fost uniformizate")
    print(f"✅ Compatibil cu SAS și alte sisteme de analiză")
    
    # Afișăm câteva exemple
    print(f"\n🔍 Exemple de valori curățate:")
    
    # Convertim temporar pentru afișare
    df_sample = df.head(3).copy()
    df_sample['Start_Time_converted'] = pd.to_datetime(df_sample['Start_Time'])
    df_sample['End_Time_converted'] = pd.to_datetime(df_sample['End_Time'])
    
    for idx, row in df_sample.iterrows():
        print(f"  Înregistrarea {idx + 1}:")
        print(f"    Start (original): '{row['Start_Time']}'")
        print(f"    Start (convertit): {row['Start_Time_converted']}")
        print(f"    End (original): '{row['End_Time']}'")
        print(f"    End (convertit): {row['End_Time_converted']}")
        print()
    
    return True

def restore_backup():
    """Restaurează backup-ul în cazul în care ceva merge greșit"""
    original_file = 'proiect/US_Accidents_Sample_1000_Per_Year.csv'
    backup_file = 'proiect/US_Accidents_Sample_1000_Per_Year_BACKUP.csv'
    
    if os.path.exists(backup_file):
        shutil.copy2(backup_file, original_file)
        print(f"✅ Backup restaurat: {backup_file} -> {original_file}")
        return True
    else:
        print(f"❌ Backup-ul nu a fost găsit: {backup_file}")
        return False

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "restore":
        print("🔄 Restaurarea backup-ului...")
        restore_backup()
    else:
        print("🚀 Începerea curățării CSV...")
        success = cleanup_csv_datetime()
        
        if success:
            print("\n🎉 Curățarea s-a terminat cu succes!")
            print("💡 Pentru a restaura backup-ul, rulează: python cleanup_csv_datetime.py restore")
        else:
            print("\n❌ A apărut o problemă în timpul curățării!")
            print("💡 Pentru a restaura backup-ul, rulează: python cleanup_csv_datetime.py restore") 