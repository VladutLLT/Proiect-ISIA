# -*- coding: utf-8 -*-
"""
Created on Fri Jan  6 12:43:53 2023

@author: vladm
"""

"""
This data was given to me by Karl Ulrich at MIT in 1986.  I didn't 
   record his description at the time, but here's his subsequent (1992) 
   recollection:
 
     "I seem to remember that the data was from a simulation of a servo
     system involving a servo amplifier, a motor, a lead screw/nut, and a
     sliding carriage of some sort.  It may have been on of the
     translational axes of a robot on the 9th floor of the AI lab.  In any
     case, the output value is almost certainly a rise time, or the time
     required for the system to respond to a step change in a position set
     point."
     "This is an interesting collection of data provided by Karl 
    Ulrich.  It covers an extremely non-linear phenomenon - predicting the 
    rise time of a servomechanism in terms of two (continuous) gain settings
    and two (discrete) choices of mechanical linkages."
"""

"""preprocesare"""
from sklearn import metrics
import pandas as pd
from sklearn import neural_network
from sklearn.model_selection import train_test_split



"""preluare informatii din fisier"""
df = pd.read_csv('servo.csv')
"""printare fisier initial pentru verificare"""
print(df)


"""date"""
xvars = df.drop(['class'], axis=1)
"""etichete"""
yvars = df[['class']]



"""impartire date si etichete in variabile train(75%-->125) si test(25%-->42)(random)""" 
xTrain, xValid, yTrain, yValid = train_test_split(xvars, yvars, test_size=0.25)

"""printare marime variabile pentru verificare"""
print(xTrain.shape, yTrain.shape, xValid.shape, yValid.shape, df.shape)



print('\nPrintare variabile train si test pentru verificare\n')

"""date train"""
print('\n', xTrain, '\n')

"""date test"""
print('\n', xValid, '\n') 

"""etichete train"""
print('\n', yTrain, '\n') 

"""etichete test"""
print('\n', yValid, '\n') 



"""Am decis sa inlocuiesc datele A, B, C, D ,E in cifre: 1, 2, 3, 4, 5"""


xTrain = xTrain.replace('A', 1)
xTrain = xTrain.replace('B', 2)
xTrain = xTrain.replace('C', 3)
xTrain = xTrain.replace('D', 4)
xTrain = xTrain.replace('E', 5)

xValid = xValid.replace('A', 1)
xValid = xValid.replace('B', 2)
xValid = xValid.replace('C', 3)
xValid = xValid.replace('D', 4)
xValid = xValid.replace('E', 5)


print('\nAfter replacing\n')

print('\n', xTrain, '\n')
print('\n', xValid, '\n')


"""realizare regresor MLP"""
regr = neural_network.MLPRegressor(hidden_layer_sizes=(10,5), activation='relu', learning_rate_init=0.01, max_iter=300)
"""antrenare"""
regr.fit(xTrain, yTrain)

"""calculare si afisarea erorii(functia loss) si a scorului regresarii r2_score pentru verificare, pentru datele train"""
mse = metrics.mean_squared_error(yTrain, regr.predict(xTrain))
rsq = metrics.r2_score(yTrain, regr.predict(xTrain))

print(rsq,mse)

"""predictii"""
y_pred = regr.predict(xValid)
y_pred = pd.DataFrame(y_pred, columns=['Predictie class'])

"""printare predictii si etichetele de validare"""
print('\n', y_pred, '\n')
yValid = yValid.rename(columns={'class':'Validare class'})
print('\n', yValid, '\n')

"""calculare si afisare scorul regresarii r2_score si a erorii patratice medii(MSE) (functia loss)"""
print('\n', metrics.r2_score(yValid, y_pred), '\n')
print('\n', metrics.mean_squared_error(yValid, y_pred), '\n')
