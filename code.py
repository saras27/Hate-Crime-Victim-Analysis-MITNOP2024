# -*- coding: utf-8 -*-
"""
Created on Fri May 31 20:20:52 2024

@author: saras
"""

#%% essential libraries
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
#%% importing data

data = pd.read_csv("Police_Department_Investigated_Hate_Crimes_20240531.csv")

print("Dimensions:")
print(data.shape)
#(1724, 23)

print(data.head(5))

#%% columns
print("Columns:")
print(data.columns)

#%% types, missing values
print("Types:")
print(data.dtypes)

print("Missing values in data:")
print(data.isna().sum())

#%% setting up dates (month and year)
data['occurence_month'] = pd.to_datetime(data['occurence_month'], format='%Y/%m/%d')

data['year'] = data['occurence_month'].dt.year
data['month'] = data['occurence_month'].dt.month

print(data.columns)
#%% choosing columns
data_columns1 = data.drop(["record_id", "occurence_month", "ncic" , 
                           "total_number_of_individual_victims", "most_serious_ucr_type", 
                           "offensive_act","suspects_ethnicity_as_a_group",
                          "data_as_of", "data_loaded_at"], axis = 1)

#%% filling in missing values for multiple bias and weapons fields

data_columns1['is_multiple_bias'] = data_columns1['is_multiple_bias'].apply(lambda x: 0 if pd.isna(x) else 1)
data_columns1["weapon_type"] = data_columns1["weapon_type"].apply(lambda x: 'Unknown' if pd.isna(x) else x)

print(data_columns1.isna().sum())

#%% renaming columns
column_rename_dict = {
    'year': 'Year',
    'month': 'Month',
    'total_number_of_victims': 'TotalVictims',
    'suspects_race_as_a_group': 'SuspectsRace',
    'total_number_of_suspects': 'TotalSuspects',
    'most_serious_ucr': 'UCR',
    'most_serious_location': 'Location',
    'weapon_type': 'Weapon',
    'most_serious_bias': 'Bias',
    'most_serious_bias_type': 'BiasType',
    'most_serious_victim_type': 'VictimType',
    'is_multiple_bias': 'IsMultipleBias',
    'total_number_of_individual_victims_adult': 'AdultVictims',
    'total_number_of_individual_victims_juvenile': 'JuvenileVictims',
    'total_number_of_suspects_adult': 'AdultSuspects',
    'total_number_of_suspects_juvenile': 'JuvenileSuspects'
}

# Use the rename method to rename the columns
data_columns1 = data_columns1.rename(columns=column_rename_dict)
print(data_columns1.columns)

#%% location
unique_location_values = data_columns1["Location"].unique()
print(unique_location_values)
print(len(unique_location_values))

#%%bias/victim type
unique_bias_values = data_columns1["Bias"].unique()
print(unique_bias_values)
print(len(unique_bias_values))


unique_biastype_values = data_columns1["BiasType"].unique()
print(unique_biastype_values)
print(len(unique_biastype_values))

unique_victype_values = data_columns1["VictimType"].unique()
print(unique_victype_values)
print(len(unique_victype_values))

#%% weapon
unique_weapon_values = data_columns1["Weapon"].unique()
print(unique_weapon_values)
print(len(unique_weapon_values))

weapon_group = {
    'Firearm (unknown whether handgun, rifle or shotgun)': 'Firearm',
    'Other gun (pellet, BB, stun gun, etc.)': 'Firearm',
    'Handgun': 'Firearm',
    'Shotgun': 'Firearm',
    'Rifle': 'Firearm'
}

data_columns1['Weapon'] = data_columns1['Weapon'].replace(weapon_group)
print(data_columns1['Weapon'].unique())
print(len(unique_weapon_values))
#%% frequency of suspects races



#%% Set the style to a chosen one
plt.style.use('seaborn-v0_8')

# Assuming you have a DataFrame called 'data' with the column 'suspects_race_as_a_group'
suspects_race_freq = data_columns1['SuspectsRace'].value_counts().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(10, 6))
suspects_race_freq.plot(kind='pie', labels=None, ax=ax)
ax.set_title("Distribution of Suspects' Races Across All Crimes")
ax.set_ylabel('')  # Remove y-label for a cleaner look

# Add legend to the side
#plt.legend(title='Race', labels=suspects_race_freq.index, loc='center left', bbox_to_anchor=(1, 0.5))

# Adjust layout to make room for the legend
#plt.tight_layout()

# Show the plot
#plt.show()

plotly_fig = mpl_to_plotly(fig)

plotly_fig.update_layout(
    width=800,
    height=600,
    margin=dict(l=50, r=50, t=50, b=50)
)

#%% distribution of number of victims

victim_counts = data_columns1['TotalVictims'].value_counts().reset_index()
victim_counts.columns = ['Number of Victims', 'Frequency']

# Sort the DataFrame by the number of victims
victim_counts = victim_counts.sort_values(by='Number of Victims')

# Display the table
print(victim_counts)

app = dash.Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

#%% dash

app = dash.Dash()
app.layout = html.Div(
    children = [ 
    html.H1("Analysis of Hate Crimes in California: Trends and Insights"),
    dbc.Row(
        [
            dbc.Col([
                dcc.Dropdown(
                    id='category',
                    value='Select category of analysis',
                    clearable=False,
                    options=["Hate Crimes Data", "Trends", "Demographic Data"]
                    ),
            ],
            #https://www.youtube.com/watch?v=dRjNfahHJRQ
            width=4
            )
        ]
    ),
    dash_table.DataTable(
        id='victim-counts-table',
        columns=[{"name": i, "id": i} for i in victim_counts.columns],
        data=victim_counts.to_dict('records'),
        style_table={'overflowX': 'scroll'},
        style_header={'backgroundColor': 'rgb(230, 230, 230)', 'fontWeight': 'bold'},
        style_cell={'textAlign': 'left', 'minWidth': '100px', 'width': '100px', 'maxWidth': '150px'},
    ),
    dcc.Graph(
            id='suspects-race-pie-chart',
            figure=plotly_fig
    )
    # dcc.Graph(
    #     id='suspects-race-plot',
    #     figure=plotly_fig
    #     ),
    # dcc.Graph(
    #     id='victim-pie-chart',
    #     figure=plotly_victim_pie_chart
    #     ),
    #]),
    ])

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)