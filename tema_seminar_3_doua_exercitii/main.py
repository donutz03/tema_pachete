import pandas as pd
#1
df = pd.read_csv('clienti_leasing.csv')

df_nou = df.loc[(df['VAL_CREDITS_RON'] == 0) & (df['DEPOSIT_AMOUNT'] > 150000),
                ['NAME_CLIENT', 'DEPOSIT_AMOUNT', 'PRESCORING']]

df_nou.loc[df_nou['DEPOSIT_AMOUNT'] > 500000, 'PRESCORING'] = 6

df_nou.to_csv('rezultat_ex1.csv', index=False)
print(df_nou)

import json
#2
with open('clienti_daune.json') as f:
    data = json.load(f)

lista_cuvinte = []
for dauna in data:
    lista_cuvinte.extend(str(dauna['Dauna']).lower().split())

dictionar = {}
for cuvant in lista_cuvinte:
    dictionar[cuvant] = dictionar.get(cuvant, 0) + 1

cuvinte_eliminate = ['the', 'and', 'to', 'a']

rezultat = []
for cuvant, frecventa in dictionar.items():
    if frecventa > 1000 and cuvant not in cuvinte_eliminate:
        rezultat.append((frecventa, cuvant))

rezultat.sort(reverse=True)
print(rezultat)