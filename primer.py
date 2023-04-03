""" PRIMER.PY
    Korte uitleg bij het gebruik van numpy arrays.
    
    Auteur: D.C.Doedens@hhs.nl
    Datum: 20220930
    
    NOOT:In verband met tekeningen in de toelichting,
    is lezen in fixed width font  aan te raden.
    
    In wiskunde zeggen we matrix; in numpy noemen we het array.
    In wiskunde zeggen we vector; in numpy noemen we het array.1
    Dus alles is een array...
    net als in de wiskunde eigenlijk alles ook als matrix gezien kan worden..
    
    
    een array van dimensie 0, ... is wel heel leeg
    een array van dimensie 1, is een reeks getallen
    een array van dimensie 2, kan een matrix of een vector zijn
"""

import numpy as np
from numpy import sin, cos, linspace, pi

""" Arrays worden opgebouwd in rijen en kolommen.
    Om ze aan te maken gebruik je (b.v.) het standaard "python datatype": list,
    waarbij elke element een rij is die uit een 'list' bestaat.
    mijnarray = np.array(
     [                             < start met een haak voor het begin van de array
                                     de array bestaat uit 3 elementen (lees: rijen)
     [element,element,element],    < Rij 0, bestaat zelf uit 3 elementen (lees: de waarden)
     [element,element,element],    < Rij 1
     [element,element,element]     < Rij 2
     ])
                        .,.,: dat is de 2e kolom
               .,:,. Dat is de 1e kolom
         :,.,. Dat is de 0e kolom
"""

#%% Vectoren
""" Een vector is een speciale vorm van een array.
    Het heeft 2 Dimensies, maar een van de dimensies is maar 1 groot.
    Sommige vectoren staan in een kolom, andere in een rij.
    De array.shape van een vector kan dus zijn:
        - (:,1) alle elementen in 1 kolom.
        - (1,:) in 1 rij alle elementen.
    
    Een reeks getallen heeft maar 1 dimensie.
    Bijvoorbeeld de getallen op een getallenlijn.
    De volgorde van de elementen heeft betekenis,
    bijvoorbeeld in de Fibionacci reeks.
    Dit is anders dan een set, maar de uitleg daarvan voert hier te ver.
"""
#%%% Vectoren creeeren
print('\n vectoren creeren:\n')

reeksgtal = np.array([ 1,2,3,4,54,65 ])  #<- let op: een set haakjes minder.
rowvector = np.array([[1,2,3,4,54,65]])
colvector = np.array([[1],[2],[3],[4]])


print(f'{reeksgtal.ndim=}\n{reeksgtal=}')
print(f'{rowvector.ndim=}\n{rowvector=}')
print(f'{colvector.ndim=}\n{colvector=}')

enen = np.ones((1,3)) # 1 rij, 3 kolommen
enen = np.ones((3,1)) # 3 rijen, 1 kolom
nullen = np.zeros((2,1)) # 2x1
leeg = np.array([[],[],[]]) # met de hand een 3x0? maken... 3 rijen, 0 kolommen (geen data)
pasopmet = np.empty((3,1))  # empty betekent hier niet leeg, maar onbepaald.
iets = np.arange(12).reshape(3,4) # 0-11 in een 3rijx4kolommen "matrix"

#%%% vectoren bewerken
print('\nbewerkingen:\n')
""" vector bewerkingen:
    Als numpy een 'bewerking' doet, dan retourneert het eigenlijk altijd een copy.
    Het ding dat 'bewerkt' wordt, blijft dus hetzelfde. Als je wat met de uitkomts wilt doen,
    dan zul je de uitkomst moeten opslaan in een andere variabele
     (of idd, dezelfde, maar dan is het overschrijven).

    a = np.array([[1],[2],[3]])
    a.T       : Transponeren, van een kolom een vector maken en omgekeerd
    a.ndim    : kun je hierboven iets van zien
    a.shape   : vorm in alle dimensies opvragen
    a.size    : aantal elementen opvragen
    help(a)   : hier kun je ook lezen over alle andere nuttige dingen
"""

print(f'{reeksgtal=}\n{reeksgtal.T=}') #weinig verschil.. geen zelfs
print(f'{rowvector=}\n{rowvector.T=}') #rowvector.T is geen rijvector meer
print(f'{colvector=}\n{colvector.T=}') #colvector.T is geen kolomvector meer

a = np.array([[1],[2],[3]])
print(a)

#%%% rekenen met vectoren
""" Vector berekeningen
    a = np.array([[1],[2],[3]])
    In de wiskunde:
        /1\
    a = |2|
        \3/
    a^T = (1,2,4)
    a*a : kan niet
    a.T*a = 1*1+2*2+3*3 = 1+4+9 = 14
    a*a.T = [[ 1*1, 1*2, 1*3], = [[1,2,3],[2,4,6],[3,6,9]]
             [ 2*1, 2*2, 2*3],
             [ 3*1, 3*2, 3*3]]
    Kan in numpy met:
    np.dot(a,a) : kan niet, gelukkig... dan lijkt het nog op de bekende wiskunde
    np.dot(a,a.T) = [[ 1*1, 1*2, 1*3], = [[1,2,3],[2,4,6],[3,6,9]]
                     [ 2*1, 2*2, 2*3],
                     [ 3*1, 3*2, 3*3]]
    np.dot(a.T,a) = [[14]]
    of met de '@' operator:
    a@a : kan nog altijd niet
    a@a.T -> "matrix"
    a.T@a -> 14 (array van 1x1)
    
    Maar let op! In numpy gaat bewerkingen hoofdzakelijk per element:
    a*a = [[a[0]*a[0],[a[1]*a[1],[a[2]*a[2]]=[[1],[4],[9]]
    a*a.T = [[a[0]*a[0],[a[0]*a[1],[a[0]*a[2], =[[1],[2],[3]
             [a[1]*a[0],[a[1]*a[1],[a[1]*a[2],   [2],[4],[6]
             [a[2]*a[0],[a[2]*a[1],[a[2]*a[2]]   [3],[6],[9]]
    a.T*a = [[a[0]*a[0],[a[1]*a[0],[a[2]*a[0], =[[1],[2],[3]
             [a[0]*a[1],[a[1]*a[1],[a[2]*a[1],   [2],[4],[6]
             [a[0]*a[2],[a[1]*a[2],[a[2]*a[2]]   [3],[6],[9]]
    
    de * operator is dus per element, en +,-,/
    de ** operator ook.
    handig, want 2*a klopt dus b.v. ook nog.
    of sin(a), de sinus van alle elementen van a.
    
"""
print('\nberekeningen:\n')
b = np.array([[2],[3],[4]])
print(f'{a=}\n{b=}')
print(f'{a.T*b=}')
print(f'{a*b.T=}')
print(f'{np.dot(a,a.T)=}')
print(f'{a@a.T=}')
print(f'{a*2=}')

if 0: #als je wel wilt plotten, hier 1 als operand van maken.
    import matplotlib.pyplot as plt #matplotlib uitleggen voert te ver voor deze primer
    t=np.linspace(0,2*pi,100) #tijd van 0 tot 2*pi in 100 stappen
    y=sin(t)
    plt.plot(t,y,'.-')
    plt.show()

#%% Matrices
""" Verschil tussen een np.array en een np.matrix is er ook niet echt,
    behalve dat de np.matrix achter de schermen wat fancy functies heeft,
    en daardoor niet helemaal compatibel is met sommige array bewerkingen.
    
    De 'anatomie' van een array staat helemaal boven beschreven,
    ook voor de vorm van een matrix.
"""
#%%% Matrices aanmaken
print('matrix aanmaken')
S = np.array([[1,2],[3,4]]) #de array lijkt op de input
A = np.arange(9).reshape((3,3))+1 #hier wordt de input ge'reshaped' en bij elk element wordt 1 opgeteld.
print(f'dit is een {A.shape} array, dus {A.shape[0]} rijen en {A.shape[1]} kolommen\nA = \n {A}')
print(A**2)
B = np.array([[1,2,3],[3,4,5]]) # deze heeft een andere vorm
O = np.zeros((3,3))   # de vorm (3,3) met nullen vullen
E = np.ones((2,2))    # de vorm (2,2) met enen vullen
F = np.full((2,2),22) # de vorm (2,2) met 22's vullen
#D = np.diagonal([1,2,3])
V = np.vstack([rowvector, rowvector]) #vectoren aan elkaar plakken
H = np.hstack([colvector, colvector]) #vectoren aan elkaar plakken
Vv = np.vstack([V, np.ones((1,rowvector.size))]) #rowvector.size geeft het aantal elementen in rowvector
Hh = np.hstack([H, np.zeros((colvector.size,1))]) #let op de (( )), zeros en ones verwachten een tuple als input

#%%% Matrices bewerken
print('matrix "bewerken"') #zoals gezegd, meestal retourneert het een kopie.
print(f'diagonaal : \n{A.diagonal(0)=}') #... dit is een functie omdat je ook diagonalen naast de hoofddiagonaal kan kiezen
print(f'diagonaal : \n{A.diagonal(1)=}') #... dit is een functie omdat je ook diagonalen boven de hoofddiagonaal kan kiezen
print(f'diagonaal : \n{A.diagonal(-1)=}') #.. dit is een functie omdat je ook diagonalen onder de hoofddiagonaal kan kiezen
print(f'eenheidsmatrix: \n{np.identity(3)}') #1 paramter, want hij hoort 4kant
print(f'matrix met enen diagonaal: \n{np.eye(3,k=1)}')# ook voor niet identiteis matrices
print(f'getransponeerd: \n{A.T}')
print(f'kolom 0: {A[:,0]=}') # lees: van kolom 0 (want het is het 2e parameter) alle elementen
print(f'rij   1: {A[-1,:]=}') # lees: van rij -1 (want het is het 1e parameter) alle elementen. standaard python telling.
                                        
r = np.array([[1],[2]]) # plaatsvector voor rx = 1, ry=2
d = np.identity(3)*(np.arange(3).T+1)
print(f'{d=}')

#%%% Rekenen met matrices
print('matrix rekenen') # en dan in combinatie met vectoren
""" Let op ook hier werkt '*' en '+' enz. elementsgewijs.
"""

print(f'{A=}\n{a=}')
print(f'{A*a=}')  # dit is dus niet wat we in wiskunde bedoelen met A*a
                  # het is de elementen van rij A[n,:] met element a[n], voor n = {0,1,2}
#print(f'{A*a.T=}') #dit kan natuurlijk niet
print(f'{a.T*A=}') #dit vermenigvuldigt elk element uit de rij van a.T, met een element uit rij(0,1,of 2) van A
                # ofwel element a[n] vermenigvuldigd met kolom A[:,n].
print(f'{np.dot(A,a)=}') # dit is wat we in de wiskunde bedoelen met A*a


""" Overigens werken kun je voor de elementen van een array ook weer lists gebruiken,
    Hiermee kun je dus 3-Dimensionale (of zelfs hogere) matrices maken.
    Leuk voor b.v. een spelletje boter kaas en eieren in 3D.
"""
Q = np.array(
    [[[1,2,3],[3,4,5],[5,6,7]],
     [[2,3,4],[4,5,6],[6,7,8]],
     [[3,4,5],[5,6,7],[8,9,9]]])
#print(f'een matrix met afmetingen {Q.shape} \n{Q=}')

#%%% Wat niet
""" NIET GEBRUIKEN
    Numpy heeft de functie, 'np.matrix()', maar gebruik die in principe NIET
    Deze heeft vooral speciale eigenschappen voor 2D matrices.
    Een numpy matrix, is een speciaal geval van een numpy array.
    Gebruik een numpy array, dat scheelt veel gedoe.
    Gebruik de numpy.matrix() *niet*, tenzij je er een specifieke toepassing voor hebt.
"""
nietgebruiken = np.matrix(np.arange(1,10).reshape((3,3)))

#%% Indices
print("indexering")
""" Elementen tellen...
    Ja, net als de standaard python begint numpy ook te tellen bij 0.
"""
C = np.arange(9).reshape(3,3)  # een 3x3 array, startend bij 0
print(f'{C=}')
print(f'{C.flat[4]=}')  # geef het 4e element, tellend: eerst langs as 1 (de rij),
                        # dan langs as 2 (de kolommen), elke rij afgaand.
for t in np.nditer(C):  # zo kun je ook alle elementen langs gaan.
    print(t, end=' ')
    
print()
print("'Slicen'")
""" Laten we met een voorbeeld:
    we vullen 'selectie' met de list [0,1,2,3,4,5]
    selectie = list(range(6))
    
    Door een gepaste 'slice' toe te passen. [{start}:{stop}:{stap}]
    Deze geef je aan met een start, stop en stapwaarde.
    Deze krijgen standaard waarden: [0:len(selectie):1]
    Met de getallen geef je het "van" tot "tot (niet met)" elementnummer op.
    Met negative indicering, is het laatse (1-laatste) element genummerd -1.
    selectie[4:6] # element 4 tot (niet met) element 6 (welke nieteens bestaat, maar dat deert niet)
    >>> [4,5]
    selectie[-6:-2] # 6-laatste element tot (niet met) het 2-laatste element.
    >>> [0,1,2,3]
    
    selectie[0:len(selectie):1] #omslachtige manier om de hele lijst op te vragen
    >>> [0,1,2,3,4,5]
    We kunnen ook negatief door de list heen stappen, dus NAAR LINKS
    selectie[::-1]
    >>> [5,4,3,2,1,0]
    En ook hier kunnen we "van element # tot (niet met) element #" opgeven, NAAR LINKS
    selectie[2:0:-1] # element 2 tot (niet met) het element 0, NAAR LINKS
    >>> [2,1] #let op dat element 0 dus niet in het resultaat zit.
    Gelukkig kun je de 0 ook weglaten
    selectie[2::-1] # element 2 tot zover er data is, naar links.
    >>> [2,1,0]
    En we kunnen hier de negative indicering gebruiken
    selectie[-4:-7:-1] # 4-laatste element tot (niet met) 7-laatste element, NAAR LINKS
    >>> [2,1,0] # zie hier dat het 7-laatste element (welke nieteens bestaat),
                # niet meegenomen wordt)
    Of een combinatie
    selectie[-2:2:-1] # van het 2-laatste tot (niet met) tot element 2, Naar links
    [4,3]
"""

#""#""#""#""#""#""#""#""#""#""#""#""#""#""#""#""#""#""#""#""#""#
""" Slices benaderen met ophak punten.
    kan helpen, maar let op bij negatieve stappen!
    Voor positieve stappen kan het helpen om de index te zien als 'ophak' punt.
    selectie[0:4] #van index 0 tot index 4
    >>> [0,1,2,3]
    selectie[0:5:2] # van index 0 tot index 5 in stappen van 2
    >>> [0,2,4]
    
    we kunnen ook indiceren van achteraf
    selectie[0:-2] # van index 0 tot het 2-voorlaatste index.
    
     0   1   2   3   4   5   6 
    -6  -5  -4  -3  -2  -1
     +---+---+---+---+---+---+
     | 0 | 1 | 2 | 3 | 4 | 5 | #slice namen, voor positieve stappen
     +---+---+---+---+---+---+
    
    
    Ook hier kunnen we de andere kant op stappen, dus NAAR LINKS.
    selectie[::-1]
    >>> [5,4,3,2,1,0]
    
    Wanneer je de indices als 'ophak' punt ziet,
    (let op!),dan krijgen de punten nieuwe namen!
    +---+---+---+---+---+---+
    | 0 | 1 | 2 | 3 | 4 | 5 | #slice namen voor negatieve stappen
    +---+---+---+---+---+---+
    -7 -6  -5  -4  -3  -2  -1
    x   0   1   2   3   4   5 # links van element 0 is index[-(len()+1)]
                              # er bestaat geen positieve index
                              # voor het element dat niet bestaat.
    
    selectie[-1:-4:-1]# de selectie van index -1 tot -4, naar links
    >>> [5,4,3]
    selectie[-2:2:-1] # de selectie van -2 tot 2 in stappen van 1 NAAR LINKS
    >>> [4,3]
    selectie[-7:3:-1] # de selectie van -7 tot 3 in stappen van 1 NAAR LINKS
    >>> []            # want we gaan naar links, dus dat levert van punt
                      # -7 naar 3 niks op.
    selectie[3:-7:-1] # selectie van 3 tot -7 in stappen van -1
    >>> [3,2,1,0]
    
"""