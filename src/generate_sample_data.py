"""
Script para generar datos de muestra para el proyecto de Boosting Algorithms
Genera archivos clean_train.csv y clean_test.csv en data/processed/
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from pathlib import Path

# Configurar la semilla para reproducibilidad
np.random.seed(42)

# Crear datos sintéticos (similar a un dataset de predicción médica)
n_samples = 768
n_features = 8

# Generar características
features = {
    'Pregnancies': np.random.randint(0, 17, n_samples),
    'Glucose': np.random.randint(44, 200, n_samples),
    'BloodPressure': np.random.randint(24, 122, n_samples),
    'SkinThickness': np.random.randint(7, 100, n_samples),
    'Insulin': np.random.randint(14, 846, n_samples),
    'BMI': np.random.uniform(18.2, 67.1, n_samples),
    'DiabetesPedigreeFunction': np.random.uniform(0.078, 2.42, n_samples),
    'Age': np.random.randint(21, 81, n_samples),
}

# Crear variable objetivo (binaria)
# Hacer que algunas características influyan en la probabilidad
X = pd.DataFrame(features)
probabilities = (
    0.3 * (X['Glucose'] > X['Glucose'].median()) +
    0.2 * (X['BMI'] > X['BMI'].median()) +
    0.15 * (X['Age'] > X['Age'].median()) +
    np.random.normal(0, 0.3, n_samples)
)
probabilities = (probabilities - probabilities.min()) / (probabilities.max() - probabilities.min())
y = (probabilities > 0.5).astype(int)

# Crear DataFrame completo
df = X.copy()
df['Outcome'] = y

# Dividir en train (80%) y test (20%)
train_data, test_data = train_test_split(df, test_size=0.2, random_state=42, stratify=y)

# Crear directorio si no existe
processed_dir = Path(__file__).parent.parent / 'data' / 'processed'
processed_dir.mkdir(parents=True, exist_ok=True)

# Guardar archivos
train_path = processed_dir / 'clean_train.csv'
test_path = processed_dir / 'clean_test.csv'

train_data.to_csv(train_path, index=False)
test_data.to_csv(test_path, index=False)

print(f"✓ Datos de entrenamiento guardados: {train_path}")
print(f"  - Filas: {len(train_data)}")
print(f"  - Columnas: {list(train_data.columns)}")
print()
print(f"✓ Datos de prueba guardados: {test_path}")
print(f"  - Filas: {len(test_data)}")
print(f"  - Columnas: {list(test_data.columns)}")
print()
print("Ahora puedes ejecutar el notebook sin errores de archivo no encontrado.")
