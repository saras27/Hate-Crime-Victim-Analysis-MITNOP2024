# -*- coding: utf-8 -*-
"""
Created on Tue Jun  4 15:43:01 2024

@author: HP
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import dash
from dash import dcc, html
import plotly.express as px
from plotly.tools import mpl_to_plotly
import dash_bootstrap_components as dbc 
import dash_table

demographic_set = pd.read_csv("San_Francisco_Population_and_Demographic_Census_data_20240531.csv")

print(demographic_set.shape)

#print(df.isna().sum())   #ne mogu da izbacim sve nedostajuce vrednosti jer bih ostala bez podataka


demographic_set = demographic_set.loc[:, ["demographic_category", "demographic_category_label", "min_age", "max_age"]]
print(demographic_set.head(5))

print("Redovi sa nedostajucom vrednoscu: \n")
print(demographic_set[demographic_set.isna().any(axis=1)])

renamed_columns = {
    'demographic_category': 'Demographic category',
    'demographic_category_label': 'Demographic category label',
    'min_age': 'Min age',
    'man_age': 'Min age'
}

demographic_set = demographic_set.rename(columns=renamed_columns)
#print(df.columns)

unique_demographic_category_values = demographic_set["Demographic category"].unique()
#print(unique_demographic_category_values)

demographic_info = demographic_set["Demographic category label"].unique()
print("Jedinstvene vrednosti za demographic_category_label: \n")
print(demographic_info)

print("Sumirano po demografskim labelama: \n")
counted_demographic_category_label_values = demographic_set["Demographic category label"].value_counts()
#print(counted_demographic_category_label_values)

#%%
# Definišemo funkciju koja izvlači rasu iz demografske informacije
def extract_race(demographic_info):
    if "Native Hawaiian or Other Pacific Islander" in demographic_info:
        return "Asian/Pacific Islander"
    elif "Two or more races" in demographic_info:
        return "Group of Multiple Races"
    elif "Some other race alone" in demographic_info:
        return "Some other race alone"
    elif "Hispanic or Latino" in demographic_info and "Not Hispanic or Latino" not in demographic_info:
        return "Hispanic"
    elif "Black or African American alone" in demographic_info:
        return "Black or African American"
    elif "American Indian or Alaska Native" in demographic_info:
        return "American Indian or Alaska Native"
    elif "Asian alone" in demographic_info:
        return "Asian"
    elif "Multi-Racial" in demographic_info:
        return "Group of Multiple Races"
    elif "White alone" in demographic_info:
        return "White"
    else:
        return "Unknown"

# Kreiramo novu kolonu "Rasa" primenom funkcije extract_race na postojeću kolonu "Demografska informacija"
demographic_set['race'] = demographic_set['Demographic category label'].apply(extract_race)
print("Rase preuredjeno: \n")
print(demographic_set["race"])

#%%
# Definišemo funkciju koja izvlači pol iz demografske informacije
def extract_sex(demographic_info):
    if "Male" in demographic_info:
        return "Male"
    elif "Female" in demographic_info:
        return "Female"
    else:
        return "Unknown"

# Kreiramo novu kolonu "Rasa" primenom funkcije extract_race na postojeću kolonu "Demografska informacija"
demographic_set['sex'] = demographic_set['Demographic category label'].apply(extract_sex)
print("Pol preuredjeno: \n")
print(demographic_set["sex"])
#%%
#prebrojavam vrednosti za race kolonu
race_counts = demographic_set["race"].value_counts().reset_index()
race_counts.columns = ["Race", "Count"]

sex_counts = demographic_set["sex"].value_counts().reset_index()
sex_counts.columns = ["Sex", "Count"]

fig_race = px.bar(race_counts, x="Race", y="Count", title="Brojnost rasnih kategorija")
fig_race.update_layout(xaxis_title="Rasa", yaxis_title="Brojnost")

#grafički prikaz plotly 
fig_sex = px.bar(sex_counts, x="Sex", y="Count", title="Brojnost muskaraca i zena")
fig_sex.update_layout(xaxis_title="Pol", yaxis_title="Brojnost")

# inicijalizacija Dash aplikacije
app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

# Definišite layout Dash aplikacije
app.layout = html.Div([
    dcc.Graph(figure=fig_race),
    dcc.Graph(figure=fig_sex) 
])

#%%
data = pd.read_csv("Police_Department_Investigated_Hate_Crimes_20240531.csv")
#print(data.columns)
#print(data["suspects_race_as_a_group"].unique())
#prazni_redovi = df[df["demographic_category"] == "gender"]
#print(prazni_redovi)
#df["demographic_category"] = df["demographic_category_label"].replace({"Female:": 0, "Male:": 1})
#print(gender_redovi)
#%%
if __name__ == '__main__':
    app.run_server(debug=True)
