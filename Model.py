import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from tkinter import Tk, filedialog
from datetime import datetime

def convertir_a_minutos(hora):
    horas, minutos = map(int, hora.split(':'))
    return horas * 60 + minutos

def cargar_archivo():
    root = Tk()
    root.withdraw()
    archivo = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
    return archivo

df = pd.read_csv("consumo_casa2.csv")
df['Fecha'] = df['Fecha'].apply(lambda x: x.split()[1])
df['Fecha'] = df['Fecha'].apply(convertir_a_minutos)

X = df[['Fecha', 'Medidor [W]']].values
Y = df[['Refrigerator', 'Clothes washer', 'Clothes Iron', 'Computer', 'Oven', 'Play', 'TV', 'Sound system']].values

X_train = X[:-1]
Y_train = Y[:-1]

model = LinearRegression()
model.fit(X_train, Y_train)

archivo_prediccion = cargar_archivo()
df_prediccion = pd.read_csv(archivo_prediccion)
df_prediccion['Fecha'] = df_prediccion['Fecha'].apply(lambda x: x.split()[1])
df_prediccion['Fecha'] = df_prediccion['Fecha'].apply(convertir_a_minutos)
X_pred = df_prediccion[['Fecha', 'Medidor [W]']].values

y_pred = model.predict(X_pred)

# Ajustar los valores negativos a 0
y_pred = np.maximum(y_pred, 0)

df_resultado = pd.DataFrame(y_pred, columns=['Refrigerator', 'Clothes washer', 'Clothes Iron', 'Computer', 'Oven', 'Play', 'TV', 'Sound system'])

# Redondear los valores a 2 decimales
df_resultado = df_resultado.round(decimals=2)

print("Predicciones para los últimos datos del archivo seleccionado:")
print(df_resultado)

score = model.score(X_train, Y_train)
print("Score del modelo:", score)

# Permitir al usuario seleccionar dónde guardar el resultado
guardar_resultado = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])

if guardar_resultado:
    df_resultado.to_csv(guardar_resultado, index=False)
    print("Resultado guardado en:", guardar_resultado)
else:
    print("No se seleccionó ningún archivo para guardar el resultado.")
