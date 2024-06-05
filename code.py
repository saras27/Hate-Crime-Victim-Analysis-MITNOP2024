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
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import ttk

plt.rcParams["axes.prop_cycle"] = plt.cycler(color = ['#aee4ff', '#6a9cfd', '#ffb8d0', '#fee5e1', '#033495'])

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

#%% about data: location
unique_location_values = data_columns1["Location"].unique()
print(unique_location_values)
print(len(unique_location_values))

#%% about data: bias/victim type
unique_bias_values = data_columns1["Bias"].unique()
print(unique_bias_values)
print(len(unique_bias_values))


unique_biastype_values = data_columns1["BiasType"].unique()
print(unique_biastype_values)
print(len(unique_biastype_values))

unique_victype_values = data_columns1["VictimType"].unique()
print(unique_victype_values)
print(len(unique_victype_values))

#%% about data: weapon
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

#%% number of victims by year

victims_by_year = data_columns1.groupby('Year')['TotalVictims'].sum().reset_index()

# Plotting the graph
victims_by_year_graph = plt.figure(figsize=(10, 6))
plt.plot(victims_by_year['Year'], victims_by_year['TotalVictims'], marker='o', linestyle='-', color='#033495')
plt.title('Total Number of Victims by Year')
plt.xlabel('Year')
plt.ylabel('Total Number of Victims')
plt.grid(True)
# plt.show()

#%% years with most and least victims by month

df = data_columns1[data_columns1['Year'] != 2024]

total_victims_by_year = df.groupby('Year')['TotalVictims'].sum()
max_year = total_victims_by_year.idxmax()
min_year = total_victims_by_year.idxmin()

max_year_data = df[df['Year'] == max_year].groupby('Month')['TotalVictims'].sum().reset_index()
min_year_data = df[df['Year'] == min_year].groupby('Month')['TotalVictims'].sum().reset_index()

fig = plt.figure(figsize=(10, 6))

# Create a subplot and plot on it
max_year_graph = fig.add_subplot(1, 2, 1)
max_year_graph.plot(max_year_data['Month'], max_year_data['TotalVictims'], marker='o', linestyle='-', color='#033495')
max_year_graph.set_title(f'Total Number of Victims by Month in {max_year}')
max_year_graph.set_xlabel('Month')
max_year_graph.set_ylabel('Total Number of Victims')
max_year_graph.grid(True)
max_year_graph.set_xticks(range(1, 13))
plt.tight_layout()
# plt.show()


#%%

# Plot for min year
min_year_graph = plt.subplot(1, 2, 2)
plt.plot(min_year_data['Month'], min_year_data['TotalVictims'], marker='o', linestyle='-', color = '#6a9cfd')
plt.title(f'Total Number of Victims by Month in {min_year}')
plt.xlabel('Month')
plt.ylabel('Total Number of Victims')
plt.grid(True)
plt.xticks(range(1, 13))

plt.tight_layout()
# plt.show()

#%% distribution of number of victims attacked at a time

victim_counts = data_columns1['TotalVictims'].value_counts().reset_index()
victim_counts.columns = ['Number of Victims', 'Frequency']

victim_counts = victim_counts.sort_values(by='Number of Victims')

print(victim_counts)
#%% 5 locations for crimes bar graph

location_mapping = {
    'Air/Bus/Train Terminal': 'Transportation',
    'Bank/Savings and Loan': 'Financial Institution',
    'Bar/Night Club': 'Entertainment Venue',
    'Church/Synagogue/Temple': 'Religious Site',
    'Commercial/Office Building': 'Commercial Building',
    'Construction Site': 'Construction Site',
    'Convenience Store': 'Retail Store',
    'Department/Discount Store': 'Retail Store',
    'Drug Store/Dr.’s Office/Hospital': 'Healthcare Facility',
    'Field/Woods/Park': 'Outdoor Area',
    'Government/Public Building': 'Government Building',
    'Grocery/Supermarket': 'Retail Store',
    'Highway/Road/Alley/Street': 'Public Space',
    'Hotel/Motel/etc.': 'Lodging',
    'Jail/Prison': 'Correctional Facility',
    'Lake/Waterway/Beach': 'Outdoor Area',
    'Liquor Store': 'Retail Store',
    'Parking Lot/Garage': 'Public Space',
    'Rental Storage Facility': 'Storage Facility',
    'Residence/Home/Driveway': 'Residential Area',
    'Restaurant': 'Food Service',
    'Service/Gas Station': 'Retail Store',
    'Specialty Store (TV, Fur, etc.)': 'Retail Store',
    'Other/Unknown': 'Other/Unknown',
    'Abandoned/Condemned Structure': 'Abandoned Structure',
    'Amusement Park': 'Entertainment Venue',
    'Arena/Stadium/Fairgrounds/Coliseum': 'Entertainment Venue',
    'ATM Separate from Bank': 'Financial Institution',
    'Auto Dealership New/Used': 'Retail Store',
    'Camp/Campground': 'Outdoor Area',
    'Daycare Facility': 'Educational Facility',
    'Dock/Wharf/Freight/Modal Terminal': 'Transportation',
    'Farm Facility': 'Agricultural Area',
    'Gambling Facility/Casino/Race Track': 'Entertainment Venue',
    'Industrial Site': 'Industrial Area',
    'Military Installation': 'Military Site',
    'Park/Playground': 'Outdoor Area',
    'Rest Area': 'Public Space',
    'School-College/University': 'Educational Facility',
    'School-Elementary/Secondary': 'Educational Facility',
    'Shelter-Mission/Homeless': 'Social Service Facility',
    'Shopping Mall': 'Retail Store',
    'Tribal Lands': 'Tribal Lands',
    'Community Center': 'Community Center'
}

data_columns1['Location'] = data_columns1['Location'].map(location_mapping)

total_victims_by_location = data_columns1.groupby('Location')['TotalVictims'].sum()

# Get the top 5 locations with the highest number of total victims
top_5_locations = total_victims_by_location.nlargest(5).reset_index()

# Plotting the graph
location_graph = plt.figure(figsize=(10, 6))
plt.bar(top_5_locations['Location'], top_5_locations['TotalVictims'])
plt.title('Top 5 Locations with Most Crimes')
plt.xlabel('Location')
plt.ylabel('Total Number of Victims')
plt.grid(axis='y')

# plt.show()




#%% race of suspects pie
suspects_race_freq = data_columns1['SuspectsRace'].value_counts().sort_values(ascending=False)
fig, ax = plt.subplots(figsize=(10, 6))
suspects_race_freq.plot(kind='pie', labels=None, ax=ax)
ax.set_title("Distribution of Suspects' Races Across All Crimes")
ax.set_ylabel('')


plt.legend(title='Race', labels=suspects_race_freq.index, loc='center left', bbox_to_anchor=(1, 0.5))

plt.tight_layout()

# plt.show()

#%% frequency of weapons used (generalized) bar

#%% most Frequent Bias 
weapon_rename_dict = {
    'Personal weapons (hands, feet, teeth, etc.)': 'Personal',
    'Unknown': 'Unknown',
    'Other (bottle, rocks, spitting)': 'Other',
    'Firearm': 'Firearm',
    'Blunt object (blugeon, club, etc.)': 'Blunt Object',
    'Knife or Other Cutting or Stabbing Instrument': 'Knife',
    'Knife or other cutting or stabbing instrument': 'Knife',
    'Vehicle': 'Vehicle',
    'Ropes or garrote strangulation or hanging': 'Strangulation',
    'Arson, fire': 'Arson'
}
#%%
df['Weapon'] = df['Weapon'].map(weapon_rename_dict)

weapon_counts = data_columns1['Weapon'].value_counts()

weapons_graph = plt.figure(figsize=(12, 8))
weapon_counts.plot(kind='bar')
plt.title('Most Frequent Weapons Used')
plt.xlabel('Weapon Type')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

#%% bias type and victim type bar and pie

bias_counts = data_columns1['BiasType'].value_counts()

bias_graph = plt.figure(figsize=(12, 8))
bias_counts.plot(kind='bar')
plt.title('Most Frequent Biases Type')
plt.xlabel('Bias Type')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.tight_layout()
#plt.show()


victim_type_counts = data_columns1['VictimType'].value_counts()

victim_type_graph = plt.figure(figsize=(10, 10))
plt.pie(victim_type_counts, labels=None, autopct='%1.1f%%', startangle=140)
plt.title('Distribution of Victim Types')
plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
plt.legend(victim_type_counts.index, title="Victim Types", bbox_to_anchor=(1, 0, 0.5, 1))
# plt.show()

#%% more detailed bias bar

bias_grouping_dict = {
    'Anti-Gay (Male)': 'Gay(Male)',
    'Anti-Transgender': 'Transgender',
    'Anti-Lesbian': 'Lesbian',
    'Anti-Lesbian/Gay/Bisexual/Transgender': 'Anti-LGBTQ',
    'Anti-Lesbian/Gay/Bisexual or Transgender (Mixed Group)': 'Anti-LGBTQ',
    'Anti-Bisexual': 'Bisexual',
    'Anti-Asian': 'Asian',
    'Anti-Black or African American': 'Black',
    'Anti-Hispanic or Latino': 'Latino',
    'Anti-White': 'White',
    'Anti-Arab': 'Arab',
    'Anti-Other Race/Ethnicity/Ancestry': 'Race',
    'Anti-Multiple Races (Group)': 'Race',
    'Anti-Jewish': 'Jewish',
    'Anti-Islamic (Muslim)': 'Islamic',
    'Anti-Catholic': 'Catholic',
    'Anti-Other Religion': 'Religion',
    'Anti-Multiple Religions (Group)': 'Religion',
    'Anti-Hindu': 'Hindu',
    'Anti-Protestant': 'Protestant',
    'Anti-Other Christian': 'Christian',
    'Anti-Mental Disability': 'Disability',
    'Anti-Physical Disability': 'Disability',
    'Anti-Gender Non-Conforming': 'Non-Binary',
    'Anti-Female': 'Female',
    'Anti-Male': 'Male',
    'Anti-Citizenship Status': 'Minority'
}

data_columns1['DetailedBias'] = data_columns1['Bias'].map(bias_grouping_dict)

broad_bias_counts = data_columns1['DetailedBias'].value_counts()

plt.figure(figsize=(12, 8))
broad_bias_counts.plot(kind='bar')
plt.title('Most Frequent Broad Biases')
plt.xlabel('Detailed Bias Type')
plt.ylabel('Frequency')
plt.xticks(rotation=45)
plt.tight_layout()
# plt.show()


# Plot for each racial group
racial_groups = ['Asian', 'Black', 'White', 'Latino']
racial_data = data_columns1[data_columns1['DetailedBias'].isin(racial_groups)]

victims_by_year_race = racial_data.groupby(['Year', 'DetailedBias'])['TotalVictims'].sum().unstack()

plt.figure(figsize=(12, 8))

for race in victims_by_year_race.columns:
    plt.plot(victims_by_year_race.index, victims_by_year_race[race], label=race)

plt.title('Number of Victims Over Time (By Racial Group)')
plt.xlabel('Year')
plt.ylabel('Number of Victims')
plt.legend(title='Racial Group')
plt.grid(True, linestyle='--', alpha=0.7)
plt.xticks(rotation=45)
plt.tight_layout()
# plt.show()
#%% juvenile victims/suspects

data_columns1.fillna(0, inplace=True)

victim_counts = data_columns1[['AdultVictims', 'JuvenileVictims']].sum()
suspect_counts = data_columns1[['AdultSuspects', 'JuvenileSuspects']].sum()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))

ax1.pie(victim_counts, labels=victim_counts.index, autopct='%1.1f%%', startangle=140)
ax1.set_title('Distribution of Victims')
ax1.axis('equal')

ax2.pie(suspect_counts, labels=suspect_counts.index, autopct='%1.1f%%', startangle=140)
ax2.set_title('Distribution of Suspects')
ax2.axis('equal')

plt.tight_layout()
# plt.show()
#%% removing unnecessary data
#%%
print(data_columns1.columns)
#%%
df = data_columns1.copy()

df['IsMultipleBias'] = df['IsMultipleBias'].fillna(df['IsMultipleBias'].mode()[0])  # Using mode for binary data
df['AdultVictims'] = df['AdultVictims'].fillna(0)  # Assuming 0 victims if missing
df['JuvenileVictims'] = df['JuvenileVictims'].fillna(0)  # Assuming 0 victims if missing
df['AdultSuspects'] = df['AdultSuspects'].fillna(0)  # Assuming 0 suspects if missing
df['JuvenileSuspects'] = df['JuvenileSuspects'].fillna(0)  # Assuming 0 suspects if 

print(df.isnull().sum())

#%% PREDICTIONS

df.sort_values(by=['Year', 'Month'], inplace=True)

import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.scatter(df.index, df['TotalVictims'], label='Total Victims')
plt.title('Total Victims over Time')
plt.xlabel('Index of Data Points (Sorted by Year and Month)')
plt.ylabel('Total Victims')
plt.legend()
plt.grid(True)
# plt.show()

from sklearn.linear_model import LinearRegression

# Prepare the data
X = np.array(df.index).reshape(-1, 1)  # Index as the independent variable
y = df['TotalVictims']

# Initialize and train the linear regression model
model = LinearRegression()
model.fit(X, y)

# Make predictions
y_pred = model.predict(X)

# Plot the linear regression line
plt.figure(figsize=(10, 6))
plt.scatter(df.index, df['TotalVictims'], label='Total Victims')
plt.plot(df.index, y_pred, linewidth=2, label='Linear Regression')
plt.title('Total Victims over Time with Linear Regression')
plt.xlabel('Index of Data Points (Sorted by Year and Month)')
plt.ylabel('Total Victims')
plt.legend()
plt.grid(True)
# plt.show()

# Print the coefficients
print(f'Intercept: {model.intercept_}')
print(f'Coefficient: {model.coef_[0]}')

#%%
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
df.sort_values(by=['Year', 'Month'], inplace=True)

# Prepare the data
X = np.array(df.index).reshape(-1, 1)  # Index as the independent variable
y = df['TotalVictims']

# Initialize and train the Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Make predictions
y_pred = model.predict(X)

# Evaluate the model
mse = mean_squared_error(y, y_pred)
print(f'Mean Squared Error: {mse}')

# Plot the results
plt.figure(figsize=(10, 6))
plt.scatter(df.index, df['TotalVictims'], label='Total Victims')
plt.plot(df.index, y_pred, linewidth=2, label='Random Forest Prediction')
plt.title('Total Victims over Time with Random Forest Regressor')
plt.xlabel('Index of Data Points (Sorted by Year and Month)')
plt.ylabel('Total Victims')
plt.legend()
plt.grid(True)
# plt.show()

#%%

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Assuming df is already loaded and sorted
df.sort_values(by=['Year', 'Month'], inplace=True)

# Prepare the data
X = np.array(df.index).reshape(-1, 1)  # Index as the independent variable
y = df['TotalVictims']

# Hyperparameter tuning using Grid Search
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

grid_search = GridSearchCV(RandomForestRegressor(random_state=42), param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X, y)

# Best parameters from Grid Search
best_params = grid_search.best_params_
print(f'Best parameters: {best_params}')

# Train the best model
best_model = grid_search.best_estimator_

# Evaluate the model using cross-validation
cv_scores = cross_val_score(best_model, X, y, cv=5, scoring='neg_mean_squared_error')
mean_cv_score = -np.mean(cv_scores)
print(f'Mean Cross-Validation MSE: {mean_cv_score}')

# Make predictions
y_pred = best_model.predict(X)

# Evaluate the model
mse = mean_squared_error(y, y_pred)
print(f'Mean Squared Error: {mse}')

# Plot the results
plt.figure(figsize=(10, 6))
plt.scatter(df.index, df['TotalVictims'], label='Total Victims')
plt.plot(df.index, y_pred, linewidth=2, label='Random Forest Prediction')
plt.title('Total Victims over Time with Improved Random Forest Regressor')
plt.xlabel('Index of Data Points (Sorted by Year and Month)')
plt.ylabel('Total Victims')
plt.legend()
plt.grid(True)
# plt.show()

#%% creating a window of tkenter app

def configure_scroll_region(event):
    canvas.configure(scrollregion=canvas.bbox("all"))

root = tk.Tk()
root.title("HateCrimesData")
root.state('zoomed')

# Create a canvas for the scrollable area
canvas = tk.Canvas(root)
canvas.pack(side="left", fill="both", expand=True)

vsb = ttk.Scrollbar(root, orient="vertical", command=canvas.yview)
vsb.pack(side="right", fill="y")
canvas.configure(yscrollcommand=vsb.set)

# Add a horizontal scrollbar to the canvas
hsb = ttk.Scrollbar(root, orient="horizontal", command=canvas.xview)
hsb.pack(side="bottom", fill="x")
canvas.configure(xscrollcommand=hsb.set)

# Create a frame inside the canvas to hold all the content
main_frame = tk.Frame(canvas)
canvas.create_window((0, 0), window=main_frame, anchor="nw")

main_frame.bind("<Configure>", configure_scroll_region)

frames = []

first_frame = tk.Frame(main_frame, bd=2, relief="groove")
first_frame.grid(row=0, column=0, padx=10, pady=10, sticky="ew")
frames.append(first_frame)

second_frame = tk.Frame(main_frame, bd=2, relief="groove")
second_frame.grid(row=1, column=0, padx=10, pady=10, sticky="ew")
frames.append(second_frame)


canvas1 = FigureCanvasTkAgg(victims_by_year_graph, master = first_frame)
canvas1.draw()
canvas1.get_tk_widget().pack(side="left", fill="none", expand="False")

# canvas2 = FigureCanvasTkAgg(max_year_graph, master = first_frame)
# canvas2.draw()
# canvas2.get_tk_widget().pack(side="left", fill="both", expand="True")

canvas3 = FigureCanvasTkAgg(location_graph, master = first_frame)
canvas3.draw()
canvas3.get_tk_widget().pack(side="left", fill="none", expand="False")

canvas4 = FigureCanvasTkAgg(weapons_graph, master = second_frame)
canvas4.draw()
canvas4.get_tk_widget().pack(side="left", fill="none", expand="False")

canvas5 = FigureCanvasTkAgg(bias_graph, master = second_frame)
canvas5.draw()
canvas5.get_tk_widget().pack(side="left", fill="none", expand="False")

canvas6 = FigureCanvasTkAgg(victim_type_graph, master = second_frame)
canvas6.draw()
canvas6.get_tk_widget().pack(side="left", fill="none", expand="False")


configure_scroll_region(None)
root.mainloop()

