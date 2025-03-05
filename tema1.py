import numpy as np
"""
1. Creează o listă de angajați (tupluri) predefinită.
"""

angajati = [(1, 'nume1','departament1',2988),
            (12, 'nume2','departament1',3319),
            (24, 'nume3','departament2',9077),
            (55, 'nume4','departament1',1233)]

"""
2. Scrie o funcție care primește lista de angajați și un nume de departament și 
returnează o listă cu angajații din acel departament.
"""
def angajati_din_departament(lista_angajati, nume_departament):
    angajati_dep = [i for i in lista_angajati if i[2]==nume_departament]
    return angajati_dep

print(angajati_din_departament(angajati, 'departament1'))

"""
3. Scrie o funcție care calculează salariul mediu al angajaților.
"""

def salariu_mediu(lista_angajati):
    salariu_mediu = np.average(list(map(lambda angajat : angajat[3], lista_angajati)))
    return salariu_mediu

print(salariu_mediu(angajati))

"""
4.Scrie o funcție care identifică angajatul cu cel mai mare salariu și cel cu cel mai mic salariu.
"""

def salariu_maxim_minim(lista_angajati):
    sal_max = max(lista_angajati, key = lambda angajat : angajat[3])
    sal_min = min(lista_angajati, key = lambda angajat : angajat[3])
    return sal_max, sal_min

sal_max, sal_min = salariu_maxim_minim(angajati)
print(sal_max[1], 'e angajatul cu salariul maxim', ', iar angajatul cu salariul minim e ', sal_min[1])

"""
5. Scrie o funcție care sortează angajații după salariu 
(de la cel mai mic la cel mai mare) și returnează o nouă listă de tupluri sortate.
"""

def sorteaza_dupa_salariu(lista_angajati):
    copie = [i for i in lista_angajati]
    copie.sort(key = lambda angajat : angajat[3], reverse=True)
    return copie

print('lista sortata descrescator dupa salariu')
print(sorteaza_dupa_salariu(angajati))