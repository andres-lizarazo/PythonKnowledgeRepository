import pandas as pd
import numpy as np
import matplotlib.pyplot as plt # or as pp
%matplotlib inline #para tener gráfico en notebook
import math
import collections
from pandas import DataFrame
from pandas import Series
from pandas import concat
from numpy.random import randn
import datetime
from dataclasses import dataclass

X=[1, 2, 3, 4]
Y=[4, 3, 2, 1]
dataFrame2 = pd.DataFrame([X,Y])
dataFrame2.head()
dataFrame2=pd.DataFrame({'uno':list(X),'dos':list(Y)},index=['a', 'b','c','d' ])
dataFrame2.head()

dict2={'edad':list(X),'peso':list(Y),}
index=['amparo', 'benito','camilo','diana']
dataFrame2=pd.DataFrame(dict2,index)
dataFrame2.head()

#iloc basándose en posición e ix basándose tanto en etiquetas como posición.  En el caso de una Serie, devuelve un único valor
# y en el caso de los DataFrame puede devolver tanto una Serie si sólo se indica la posición de fila, o un valor único si se indican fila y columna.
dict3= ['Casillas', 'Ramos', 'Pique', 'Puyol', 'Capdevila', 'Xabi Alonso', 'Busquets', 'Xavi Hernandez', 'Pedrito', 'Iniesta', 'Villa']

index3=[1, 15, 3, 5, 11, 14, 16, 8, 18, 6, 7]
serie= pd.Series(dict3, index3)
spanishPlayers = pd.Series(dict3, index3)
print ("Spanish Football Players: \n%s" % spanishPlayers)

dict4= ['Casillas', 'Ramos', 'Pique', 'Puyol', 'Capdevila', 'Xabi Alonso', 'Busquets',
        'Xavi Hernandez', 'Pedrito', 'Iniesta', 'Villa']
index4=['1', '15',' 3',' 5',' 11', '14', '16', '8', '18',' 6',' 7']
dataFrame4= pd.DataFrame({'nombre':list(dict4),'camisa': list(index4)})
spanishPlayers = pd.DataFrame(dict4, index4)
dataFrame4.head()
dataFrame5=pd.DataFrame([dict4,index4],{'nombre':list(dict4),'camisa': list(index4)})
dataFrame5.head()

spanishPlayersDF = pd.DataFrame(
    {
        'name': ['Casillas', 'Ramos', 'Pique', 'Puyol', 'Capdevila', 'Xabi Alonso', 'Busquets', 'Xavi Hernandez',
                 'Pedrito', 'Iniesta', 'Villa'],
        'demarcation': ['Goalkeeper', 'Right back', 'Centre-back', 'Centre-back', 'Left back', 'Defensive midfield',
                        'Defensive midfield', 'Midfielder', 'Left winger', 'Right winger', 'Centre forward'],
        'team': ['Real Madrid', 'Real Madrid', 'FC Barcelona', 'FC Barcelona', 'Villareal', 'Real Madrid',
                 'FC Barcelona', 'FC Barcelona', 'FC Barcelona', 'FC Barcelona', 'FC Barcelona']
    }, columns=['name', 'demarcation', 'team'], index=[1, 15, 3, 5, 11, 14, 16, 8, 18, 6, 7]
)
spanishPlayersDF.head()

spanishPlayersDF.loc[1] = ['Cesc', 'Forward', 'Arsenal']
spanishPlayersDF.head()

spanishPlayersDF.iloc[2]

dictNombre=['Casillas', 'Ramos', 'Pique', 'Puyol', 'Capdevila', 'Xabi Alonso', 'Busquets', 'Xavi Hernandez',
            'Pedrito', 'Iniesta', 'Villa']

dictDemarca=['Goalkeeper', 'Right back', 'Centre-back', 'Centre-back', 'Left back', 'Defensive midfield',
                        'Defensive midfield', 'Midfielder', 'Left winger', 'Right winger', 'Centre forward']

dictEquipo= ['Real Madrid', 'Real Madrid', 'FC Barcelona', 'FC Barcelona', 'Villareal', 'Real Madrid',
                 'FC Barcelona', 'FC Barcelona', 'FC Barcelona', 'FC Barcelona', 'FC Barcelona']

index=[1, 15, 3, 5, 11, 14, 16, 8, 18, 6, 7]

spanishPlayers=pd.DataFrame({'Nombre':list(dictNombre),'Posición':list(dictDemarca),
                             'Equipo': list(dictEquipo), 'camisa':list(index) } )
spanishPlayers.head()

spanishPlayers=pd.DataFrame([dictNombre,dictDemarca,dictEquipo, index],
                            {'Nombre':list(dictNombre),'Posición':list(dictDemarca),
                             'Equipo': list(dictEquipo), 'camisa':list(index),})
spanishPlayers.head()

spanishPlayers.loc['hijos'] = [1, 4,2,3,4,1,2,3,4,1,2]
spanishPlayers.head()

#NUEVO EJEMPLO

raw_data = {'regiment': ['Nighthawks', 'Nighthawks', 'Nighthawks', 'Nighthawks', 'Dragoons', 'Dragoons', 'Dragoons', 'Dragoons', 'Scouts', 'Scouts', 'Scouts', 'Scouts'],
            'company': ['1st', '1st', '2nd', '2nd', '1st', '1st', '2nd', '2nd','1st', '1st', '2nd', '2nd'],
            'deaths': [523, 52, 25, 616, 43, 234, 523, 62, 62, 73, 37, 35],
            'battles': [5, 42, 2, 2, 4, 7, 8, 3, 4, 7, 8, 9],
            'size': [1045, 957, 1099, 1400, 1592, 1006, 987, 849, 973, 1005, 1099, 1523],
            'veterans': [1, 5, 62, 26, 73, 37, 949, 48, 48, 435, 63, 345],
            'readiness': [1, 2, 3, 3, 2, 1, 2, 3, 2, 1, 2, 3],
            'armored': [1, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1],
            'deserters': [4, 24, 31, 2, 3, 4, 24, 31, 2, 3, 2, 3],
            'origin': ['Arizona', 'California', 'Texas', 'Florida', 'Maine', 'Iowa', 'Alaska', 'Washington', 'Oregon', 'Wyoming', 'Louisana', 'Georgia']}

df = pd.DataFrame(raw_data, columns = ['regiment', 'company', 'deaths', 'battles', 'size', 'veterans', 'readiness', 'armored', 'deserters', 'origin'])

df = df.set_index('origin')

df.head()

df[['size', 'veterans']] #TABLA CON SOLO ESTAS COLUMNAS
df.loc[:'Arizona']# FILA CON TODAS LAS COLUMNAS


df.iloc[:2]  # Select every row hasta la 3era

# Select the 1 y 2 row
df.iloc[1:2]

# Select every row after the 2nd row
df.iloc[2:]

# Select the first 2 columns
df.iloc[:,:2]

# Select rows where df.deaths is greater than 50
df[df['deaths'] > 50]
#simultaneo
# data[(data['rnd_2'] == 2) & (data['rnd_3'] > 15)]

# Select rows where df.deaths is greater than 500 or less than 50
df[(df['deaths'] > 500) | (df['deaths'] < 50)]

# Select all the regiments not named "Dragoons"
df[~(df['regiment'] == 'Dragoons')]

# .ix is the combination of both .loc and .iloc. Integers are first considered labels, but if not found, falls back on positional indexing

df.ix[['Arizona', 'Texas']]
df.ix['Arizona', 'deaths'] #muertes en Arizona
df.ix[2, 'deaths'] # tercera fila muertes

# .iloc[row,column]
#from pandas import Series
serie = Series(['a', 'b', 'c'])
serie


df1 = DataFrame(randn(4,2), index=['a', 'b', 'c', 'd'], columns=['c1', 'c2']) #DataFrame con valores aleatorios

concat([df1, df2], axis=0) # concatena 2 tablas una bajo la otra
concat([df1, df2], axis=1) # concatena 2 tablas una al lado de la otra

df.melt(id_vars=['regiment','company', 'deaths','battles'], var_name='quantity') # pivotear alguna Columna2

dft = pd.DataFrame({"A1970" : {0 : "a", 1 : "b", 2 : "c"},
                     "A1980" : {0 : "d", 1 : "e", 2 : "f"},
                     "B1970" : {0 : 2.5, 1 : 1.2, 2 : .7},
                     "B1980" : {0 : 3.2, 1 : 1.3, 2 : .1},
                    "X"     : dict(zip(range(3), np.random.randn(3)))
                    })

dft["id"] = dft.index #CREACION DE UNA PRIMARY KEY
pd.wide_to_long(dft, ["A", "B"], i="id", j="year") #MIRAR XD

df.stack()
df.unstack()

df = pd.DataFrame({'A': ['one', 'one', 'two', 'three'] * 6,
   ....:                    'B': ['A', 'B', 'C'] * 8,
   ....:                    'C': ['foo', 'foo', 'foo', 'bar', 'bar', 'bar'] * 4,
   ....:                    'D': np.random.randn(24),
   ....:                    'E': np.random.randn(24),
   ....:                    'F': [datetime.datetime(2013, i, 1) for i in range(1, 13)] +
   ....:                         [datetime.datetime(2013, i, 15) for i in range(1, 13)]})


pd.pivot_table(df, values='D', index=['A', 'B'], columns=['C']) #pivot table de la anterior


#datos del 2 al 4 en el índice y las columnas 'rnd_2' y fecha
data.loc[2:4, ['rnd_2', 'fecha']]

# poner columna como index CLAVE LOGICA
data_fecha = data.set_index('fecha')
data_fecha.head()

#seleccionar casilla det
data_fecha.loc[fecha_1,'rnd_1']

# Se seleccionan las columnas rnd_2 y rnd_3 y se imprimen los primeros 5 datos
# En este caso los dos puntos quiere decir que se seleccionan todas las filas
data_fecha.iloc[:,[1,2]].head()

#Se seleccionan las filas 1, 12, 34 y las columnas 0 y 2
data_fecha.iloc[[1,12,34],[0,2]]

#reordena
df.stack('tipo')
# Ahora "tipo" como una clasificación interna de "Rh"
df.stack(level=['Rh','tipo'])


#lectura de archivo
Tabla_archivo = pd.read_excel('archivo.xlsx', index=False)
DF_tabla_archivo= pd.DataFrame(Tabla_archivo[['Columna1', 'Columan2', 'Columna3']])

Tabla2= pd.pivot_table(Tabla_archivo, values='ColumnadeValores', index=['ColumnaCategoria', 'ColumnaSubcategoria'])

Tabla2.loc['Categoria15','Subcategoria34'] # trae el valor que se encuentra en columna de valores para la especificacion en otras ColumnadeValores

Tabla_archivo.head() #muestra primeras 5 filas de la Tabla_archivo


variable = hoja_excel.loc[:,'Columnax']
ArrayVariable=np.array(variable)
rango = np.max(ArrayVariable)- np.min(ArrayVariable)
Cant_Rangos = 6
anchoRangos = rango/Cant_Rangos*1.0

bins_limits = np.linspace(np.min(ArrayVariable)-0.5, np.max(ArrayVariable)+0.5, 6)

binplace = np.digitize(edad, bins_limits)
print (binplace)
bin_counts = [len(ArrayVariable[binplace == i]) for i in range(1, len(bins_limits))]
print (bin_counts)
freq,bins=np.histogram(ArrayVariable,bins_limits)
print(freq)
print(bins)

weigth=np.ones(len(ArrayVariable))
table_freq=pd.crosstab(binplace,weigth)



plt.hist(hoja_excel['ColumnadeValores'],bins=bins_limits,color='blue', edgecolor="white")
plt.plot() #histograma

intervals_name=[]
for i in range(len(bins)-1):
    lim_inf=bins_limits[i]
    lim_sup=bins_limits[i+1]
    label='['+str(lim_inf)+","+str(lim_sup)+")"
    intervals_name.append(label)
    print(label)

freq_table=pd.crosstab(binplace,np.ones(len(edad)))
freq_table
freq_table.index=(intervals_name)
freq_table.columns=(['Frequencia'])
freq_table

%matplotlib inline

var2 = hoja_excel.loc[:,'Columna4']
ArrayVar2= np.array(var2)

DF_tabla_archivo.boxplot('Columna2')
DF_tabla_archivo.boxplot('Columan4', by='Columna5')

DF_tabla_archivo.describe()# Aqui utilizamos la funcion "describe" para calcular la media, mediana, desviación estándar de las variables



##RANDOM VALUES
X1 = np.random.normal(20, 5, 500) # Genera 500 valores para la variable aleatoria normal
X2 = np.random.uniform(low=15, high=25, size=500) # Genera 500 valores para la variable aleatoria uniforme con sus limites dados
np.mean(X1) # media
np.std(X1) #desvesta
np.var(X2)
a = 15
b = 25
var_unif=((b-a)*(b-a))/12 #varianza de la distribucióm uniforme que está dado por (b-a)^2/12 donde a=15 y b=25

#TABLA CON LAS VARIABLES CREADAS
tabla = pd.DataFrame({'Normal':list(X1) , 'Uniforme':list(X2)})

pob_0=10080734 #poblacion inicial
n=10 #años
pob_i=pob_0
k=0.025 #2.5% de crecimiento anual
pob_var=[] # lista para valores
n_var=[] # lista para periodos

for i in range (n):
    pob_i= pob_i*(1+k)
    pob_var.append(pob_i)
    n_var.append(i+1)
    #print((i+1)," ",x_i)

plp.plot(n_var,pob_var,linewidth=2,color="red")

#LOGISTIC MAP
formula ='𝑋𝑛+1=𝑟𝑋𝑛(1−𝑋𝑛)'

from pylab import show, scatter, xlim, ylim
from random import randint
iter = 1000         # Numero iteraciones por punto
x_0 = 0.5          # valor inicial x in (0, 1)
spacing = .0001     # difernecias entre puntos dominio (r-axis)
res = 10           # n-circulo mas grande visible
# listas de r y x
rlist = []
xlist = []
def logisticmap(x, r):
    return x * r * (1 - x)
def iterate(n, x, r):
    for i in range(1,n):
        x = logisticmap(x, r)
    return x   # devuelve enumero iteraciones del logistic map(x. r)
for r in [i * spacing for i in range(int(1/spacing),int(4/spacing))]:
    rlist.append(r)
    xlist.append(iterate(randint(iter-res/2,iter+res/2), x_0, r))# Generate list values -- iterate for each value of r
scatter(rlist, xlist, s = .01,color='red')
xlim(0.9, 4.1)
ylim(-0.1,1.1)
show()

#DISCRETE LOGISTIC EQUATION WITH PARAMETER r
def f(x, r):
    """Discrete logistic equation with parameter r"""
    return r*x*(1-x)

if __name__ == '__main__':
    # initial condition for x
    ys = []
    rs = np.linspace(0, 4, 400)
    for r in rs:
        x = 0.1
        for i in range(500):
            x = f(x, r)

        for i in range(50):
            x = f(x, r)
            ys.append([r, x])

    ys = np.array(ys)
    pylab.plot(ys[:,0], ys[:,1], '.')
    pylab.show()

# 3ra forma - no idea
for i in range(50):
    def f(x, r):#EC.logistica con parametro r
        return(r*x*(1-x))

if __name__ == '__main__':#  condicion inicial para x

    ys = []
    rs = np.linspace(0, 4, 400)

    for r in rs:
        x = 0.1 # se asignan los valores en rs uno a la vez

        for i in range(500):# hacerlo 500 veces.
            x = f(x, r) # Evalúa f en (x, r). valor retornado asignado a x.
                         # x es realimentada a  f(x, r). Cambia 500 veces segun ecuacion logistica

        for i in range(50):# hacerlo 50 veces
            x = f(x, r)
            ys.append([r, x]) # Save the point (r, x) in the list ys

    # ys is a list of lists.
    # You can also think of ys as a list of [r, x] point.
    # This converts the list of lists into a 2D numpy array.
    ys = np.array(ys)
    pylab.plot(ys[:,0], ys[:,1], '.')  # ys[:,0] is a 1D array of r values, ys[:, 1] is a 1D array of x values
    pylab.show()

#RELACION AUREA
def fib(n):
    a = 1
    b = 0
    for i in range(n):
        a, b = b, a + b
    return b
def fi(n):
    return fib(n) / fib(n - 1)
print('φ =', fi(100))


def fibonacci(n):
    result = []
    x, y = 0, 1
    while x < n:
        result.append(x)
    x, y = y, y + x
    print (result)
fibonacci(1000)


from math import sqrt
fi = 1.618033988749895
def fib(n):
    return (fi ** n - (1 - fi) ** n) / sqrt(5)
for i in range(16):
    print("fib({0}) = {1}".format(i, fib(i))) #ELEMENTO N-EESIMO SUCESION

from turtle import *# funciones de gráficas DE TURTLE
fi = 1.618033988749895
def espiral(n):
    radio = 10 #  radio inicial

    for i in range(n):    # n veces.
        circle(radio,90) # Dibujar el arco de un cierto radio Y  90 grados.
        radio *= fi       # Aumentar radio por el factor fi.
hideturtle()       # Esconder
color('SteelBlue') # color de la pluma: azul metálico.
pensize(20)        # tamaño de  pluma: 20 pixeles.
speed('fastest')   # Establecer velocidad de tortuga: máxima.
setheading(270)    # Apuntar la tortuga hacia el sur.
espiral(8)         # Dibujar espiral de orden 8.
done()             # Evitar  ventana cierre


def cuadro(longitud): #cuadro de lado longitud

    begin_fill()
    for i in range(4):
        fd(longitud)
        lt(90)
    end_fill()

def rectangulo_dorado(n, longitud):
    #rectengulo de n cuadrados. El primer cuadrado tiene sus lados de tamaño 'longitud'.

    for i in range(n):
        cuadro(longitud)

        # Mover la tortuga a la esquina contraria y girarla 90 grados a la izquierda.
        fd(longitud)
        lt(90)
        fd(longitud)

        longitud *= fi

def espiral_dorada(n, radio):
    #Dibuja una espiral dorada compuesta de 'n' arcos. El primer arco tiene un radio de tamaño 'radio'.

    for i in range(n):
        circle(radio, 90)
        radio *= fi

def espiral(n, radio):
    #Dibuja una espiral dorada de orden 'n' sobre el rectángulo dorado que lo alberga.
    #El primer arco/cuadro tiene un radio/lado de tamaño 'radio'.


    color('White')
    fillcolor('SteelBlue')
    pensize(1)
    rectangulo_dorado(n, radio)

    # Regresa la tortuga al centro de la pantalla pero sin dibujar por su paso.
    penup()
    home()
    pendown()

    color('Gold')
    pensize(4)
    espiral_dorada(n, radio)

hideturtle()
speed('slow')
espiral(11, 2)
done()

%lsmagic

%%latex
$\begin{center}
Analisis exploratorio de datos
\end {center}
\begin {itemize}
items medidas de tendencia central
\end {itemize}$
$$\bar{x}$$


np.percentile(variable,[25,50,75])




#list = []
#tuple =()
# dict ={}

list =[1,2,3,4]
list.insert(position, Listobject)
del list[position]

list.remove(Listobject)
list.sort()
newlist = sorted(list, reverse = True)
list[0:2] # primer y segundo ELEMENTO
list[:4] # del primer al cuarto ELEMENTO
list[3:] # del cuarto al ultimo
list[:] # todo, copia de la lista
list[0:7:2] # del primero al septimo intercalado
list[-3:-1] # del 3 de atras para adelante si incluir el primero de atrásc
list[2:4] = [x,y] # reasigna ColumnadeValores
del list[4:6] # borra valores



len([]) #largo de lista vacia =0

tuple= (1,2,3,4) # can do indexing, slicing but not modifying

#unpackaging tuple with *
def three_args(a,b,c):
    print(a,b,c)

my_args = (1,2,3)
three_args(*my_args) # devuelve 1 2 3



#DICTIONARIES AND SETS

capitals = {'US': 'Washington, DC' , 'France': 'Paris', 'Italy': 'Rome'} #creación
capitals['Italy'] # traer el atributo con la key
capitals['Spain'] = 'Madrid' #añade objeto al diccionario
'Germany' in capitals, 'Italy' in capitals # devuelve (False, True) verificando existencia en diccionario

moreCapitals = {'Germany': 'Berlin', 'UK': 'London'}
capitals.update(moreCapitals) #combina los diccionarios

del capitals('UK') #borrar item con su key
# LAS KEYS NO NECESARIAMENTE STRINGS, SINO CUALQUIER HASHABLE (CONVERTIBLE A NUMERO)
#por ejemplo un tuple
birthdays = {(7,15):'Michele', (3,14):'Albert'}
#se comprueba con HASH
hash('Italy')
hash((7,15))

#LOOPS EN diccionario

for key in dictionary: # loop over the keys

for key in dictionary.keys(): #loop explicitly over the keys

for value in dictionary.values(): #Loop over the VALUES

for key, value in dictionary.items(): #loop over key and value together

for country in capitals.keys():#OVER KEYS
    print(country)

capitals.keys() #special iterator Listobject
#pasarlo a lista
list(capitals.keys())

for capital in capitals.values():#OVER VALUES
    print(capital)

for country, capital in capitals.items()#OVER KEYS AND VALUES
    print(country, capital) #tuple unpackaging

#SETS, BAGS OF ITEMS MIXET TYPES AND WITHOUT DUPLICATIONS, NOT INDEXED

continents = {'Africa','America','Europe','Asia','Oceania', 'Africa'} # si tenemos duplicados, solo nos mostrará 1 vez cada uno

'Africa' in continents #verificar existencia en set

continents.add('Antarctica') #añade al set
continents.remove('Antarctica') #elimina del set

for c in continents: #loop over the set
    print (c)

#Metodos y atributos de clases son almaacenados internamente en los diccionarios


#Iterate on lists or Dicts
#colecto results on new list or dicts

squares=[]
for i in range (1,11):
    squares.append(i**2)
#COMPREHENSIONS: Compressed version of Loop
squares = [i**2 for i in range(1,11)] #computation before the loop inside a list

#filter inside COMPREHENSIONS
squares_by_four = [i**2 for i in range(1,11) if i**2 % 4 == 0] # condicion de que sea divisible por 4
# devuelve  [4,16,36,64,100]
#replace map and filter built.in functions

#DICTIONARY COMPREHENSIONS
squares_dict = {i: i**2 for i in range (1, 11)}
#used sometimes to transpose an existing dict, change keys

# GENERATOR EXPRESSIONS
sum(i**2 for i in range(1,11))
#naked comprehensions for generate sequences and consume elements one by one
#without ever storing them in a list or dict
#in this example fed directly to sum (saving memory and time)

#NESTED LOOP
counting = []
for i in range(1,11):
    for j in range(1, i+1):
        counting.append(j)
print(counting)

#lo mismo que este NESTED COMPREHENSIONS
counting = [j for i in range (1,11) for j in range (1, i+1)]
print(counting)
# imprime lista [1,1,2,1,2,3,1,2,3,4,.....1,2,3,4,5,6,7,8,9,10]


#ADVANCED CONTAINERS

people = [
    ("Michele", "Vallisneri", "July 15"),
    ("Albert", "Einstein", "March 14"),
    ("John", "Lennon", "October 9"),
    ("Jocelyn", "Bell Burnell", "July 15")
]

people [0][0]
# 'Michele'
people [0][1]
# 'Vallisneri'

[person for person in people if person[2] == "July 15"] #loop sobre los que tienen ese valor

#SPECIALIZED TUPLE

persontype = sollections.namedtuple('person', ['firstname', 'lastname', 'birthday' ]) #name and asociate labels with fields

#Instances of the namedtuple
michele = persontype("Michele","Vallisneri","July 15")
#si se usan los fiel names, se puede switchear
michele = persontype(lastname="Vallisneri",firstname="Michele",birthday="July 15")

#aun funcionan indices...
michele[0], michele[1], michele[2]
#pero mejor usar sintaxis orientada a objetos
michele.firstname, michele.lastname, michele.birthday

persontype(people[0])# Incorrecto usar este, no hacer Feed a tuple directly to persontype

#TUPLE unpackaging
persontype(*people[0])

#Completo a partir de list comprehensions
namedpeople = [persontype(*person) for person in people]
namedpeople # imprime la lista con los persontype de todos los que estaban en la lista de tuplas "people"

#Birthsday search improved
[person for person in namedpeople if person.birthday == "July 15"]

#Python 3.7 alternative to tuples for storing data records DATACLASSES

@dataclass
class personclass:
    firstname: str
    lastname: str
    birthday: str = 'unknown'

michele = personclass('Michele', 'Vallinsneri')
michele

michele = personclass(firstname ='Michele', lastname='Vallisneri')
michele.firstname, michele.lastname, michele.birthday #Se puede acceder el campo por nombre pero no por indice
michele[0] #ERROR

#DEFINE METHODS THART OPERATE ON THE FIELDS, EXAMPLE METHOD TO RETURN FULLNAME
@dataclass
class personclass2:
    firstname: str
    lastname: str
    birthday: str = 'unknown'

    def fullname(self):
        return self.firstname+' '+self.lastname

michele = personclass2('Michele','Vallisneri', 'July 15')
michele.fullname()

#FETURES OF DATA DATACLASSES
--
--
--
#LEARN MORE

#COLLECTIONS DEFAULT DICT

#esc + a para insertar celda arriba en notebooks
def mydefault():
    return "I don't know"

questions = collections.defaultdict(mydefault)
questions['The meaning of life'] #asknig for an unexisting key return the default and makes it part of the dict
#useful for creating dict from lists
