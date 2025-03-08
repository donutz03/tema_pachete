comenzi_lista = [[1,'nume1', ['uscator','mar','masina'],2377,"Finalizata"],
           [15,'nume2', ['para','fixativ','masina de spalat', 'stup'],9999,"Anulata"],
            [23,'nume3', ['calculator','monitor'],1766,"In curs"],
            [88,'nume4', ['mouse', 'tastatura','birou', 'scaun'],72377,"Finalizata"],
            [11,'nume5', ['unitate','tabla','marker', 'pizza', 'cafea'],6682,"In curs"]
           ]

statusuri = ("Finalizata", "Anulata", "In curs")

"""
Extrage toate comenzile care au statusul "Finalizată".
"""
def extrage_finalizate(comenzi):
    finalizate = [comanda for comanda in comenzi if comanda[4]=="Finalizata"]
    return finalizate

"""
Calculează suma totală a comenzilor care sunt "Finalizate".
"""
def suma_finalizate(comenzi):
    suma = 0
    for comanda in comenzi:
        if comanda[4]=='Finalizata':
            suma += comanda[3]
    return suma

"""
Găsește comanda cu cel mai mare total de plată.
"""
def maxim_de_plata(comenzi):
    maxim = 0
    for comanda in comenzi:
        if comanda[3] > maxim:
            maxim = comanda[3]
    return maxim

"""
Sortează comenzile în funcție de totalul comenzii, în ordine crescătoare.
"""
def sorteaza_comenzi_dupa_total_descrescator(comenzi):
    lista_sortata_desc = [i for i in comenzi]
    lista_sortata_desc.sort(key= lambda element : element[3], reverse=True)
    return lista_sortata_desc

"""
Schimbă statusul comenzii cu un anumit ID (de exemplu, ID-ul 3) la "Anulată".
"""
def schimba_status_comanda(id_comanda, status_nou, comenzi):
    comanda_de_schimbat = []
    if status_nou not in statusuri:
        return "status incorect"
    else:
        for comanda in comenzi:
            if comanda[0] == id_comanda:
                comanda[4] = status_nou
                comanda_de_schimbat = comanda
                break
    return comanda_de_schimbat

"""
Adaugă o nouă comandă în lista de comenzi.
"""
def adauga_comanda_in_lista(comanda, comenzi):
    if isinstance(comanda, list):
        if len(comanda) == 5:
            comenzi.append(comanda)
        else:
            return "comanda trebuie sa aiba 5 campuri"
    else:
        return "comanda trebuie sa fie o lista"
    return comenzi[len(comenzi)-1]

print(extrage_finalizate(comenzi_lista))
print(suma_finalizate(comenzi_lista))
print(maxim_de_plata(comenzi_lista))
print(sorteaza_comenzi_dupa_total_descrescator(comenzi_lista))
print(schimba_status_comanda(2, "lalala", comenzi_lista))
print(schimba_status_comanda(1, "Anulata", comenzi_lista))
print(adauga_comanda_in_lista([23,'nume nou de adaugat', ['obiect nou', 'alt ob', 'inca un ob'],222, 'In curs'], comenzi_lista))

print("comenzi actualizate\n\n", comenzi_lista)