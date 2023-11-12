import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

df = pd.read_csv('https://drive.google.com/u/0/uc?id=1weaiUCwhCR35QD9qVzHiDODe4R8tQMoW&export=download')
df_gdp = pd.read_csv("https://drive.google.com/u/0/uc?id=1aNIPh-dVyuUMWTgQpv5A0oxbwvJVoLrY&export=download", sep=';')
df_population = pd.read_csv("https://drive.google.com/uc?id=1XgOicjwCg47sulhhtEIZ0P981ZuGg_r7&export=download")

cols_to_remove: list[str] = []

for col in df:

  cantidad_nulos = df[col].isnull().sum()

  if cantidad_nulos < 500:
    continue

  cols_to_remove.append(col)

# Las columnas 'Code', 'Executions' y 'Terrorism' poseen demasiados nulos para ser consideradas en el analisis


df = df.drop(columns=cols_to_remove)

df['GDP'] = np.nan
df['Population'] = np.nan

# Buscamos en el DF de poblacion los datos que necesitamos
for index, row in df.iterrows():
  country, year = [str(e) for e in row[['Entity', 'Year']]]

  gdp_country_df = df_gdp[df_gdp['Country Name'].str.match(country, case=False, na=False)]
  population_country_df = df_population[df_population['Country Name'].str.match(country, case=False, na=False)]

  # Si no se encuentra la region en el dataset, ignorar
  if not population_country_df[year].any():
    continue

  gdp_year = year
  if not gdp_country_df[year].any():
    for yr in range(int(year), 2023):
      if gdp_country_df[str(yr)].any():
        gdp_year = str(yr)
        break
    continue

  gdp = gdp_country_df[gdp_year].iloc[0]
  population = population_country_df[year].iloc[0]

  if population:
    df.at[index, 'Population'] = population
  if gdp:
    df.at[index, 'GDP'] = gdp

df = df.dropna()

new_names = {}
for old_name in df.columns:
  if "Deaths" not in old_name:
    continue

  new_name = old_name.split(" - ")[1]

  new_names[old_name] = new_name

df = df.rename(columns=new_names)

for col in df.columns:
  if col in ('Entity', 'Year', 'Population'):
    continue
  df[col] = df[col] * 100_000_000 / df['Population']  # Multiplica por 100.000.000 para prevenir errores de coma flotante, despues se van a escalar

df.drop('Population', axis=1, inplace=True)

categorical_columns = "Entity-Year".split('-')

scaler = MinMaxScaler(feature_range=(0, 100))

for col in df.columns:
  if col in categorical_columns:
    continue

  df[col] = scaler.fit_transform(df[[col]])

categorical_columns = "Entity-Year".split('-')

for col in categorical_columns:
  dummies = pd.get_dummies(df[[col]], prefix=col)
  df = pd.concat([df, dummies], axis=1)
  df = df.drop([col], axis=1)

df.to_csv('sandbox/data/processed_data.csv', sep=',', index=False, encoding='utf-8')
