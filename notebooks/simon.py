import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
import plotly.express as px

matplotlib.use('Agg')  # Use the 'Agg' backend for saving figures
import warnings
warnings.filterwarnings('ignore')
#st.set_option('deprecation.showPyplotGlobalUse', False)


# Configure Streamlit
st.set_page_config(page_title="EA Sports Team Analysis", page_icon="⚽", layout="wide")

# Load the data
teams_data = pd.read_csv('teams_data.csv')

# Remove unnecessary columns
selected_columns = ['team_id', 'team_url', 'fifa_version', 'fifa_update', 'update_as_of',
       'team_name', 'league_id', 'league_name', 'league_level',
       'nationality_id', 'nationality_name', 'overall', 'attack', 'midfield',
       'defence', 'coach_id', 'home_stadium', 'rival_team',
       'international_prestige', 'domestic_prestige', 'transfer_budget_eur',
       'club_worth_eur', 'starting_xi_average_age', 'whole_team_average_age',
       'captain', 'short_free_kick', 'long_free_kick', 'left_short_free_kick',
       'right_short_free_kick', 'penalties', 'left_corner', 'right_corner',
       'def_style', 'def_team_width', 'def_team_depth', 'def_defence_pressure',
       'def_defence_aggression', 'def_defence_width',
       'def_defence_defender_line', 'off_style', 'off_build_up_play',
       'off_chance_creation', 'off_team_width', 'off_players_in_box',
       'off_corners', 'off_free_kicks', 'build_up_play_speed',
       'build_up_play_dribbling', 'build_up_play_passing',
       'build_up_play_positioning', 'chance_creation_passing',
       'chance_creation_crossing', 'chance_creation_shooting',
       'chance_creation_positioning']
teams_data = teams_data[selected_columns]

#taking out international friendlies 
teams_data = teams_data[teams_data['league_name']!='Friendly International']

##selecting top tier leagues only
teams_data = teams_data[teams_data['league_level']=='1']

# Main page title and description
st.title("EA Sports Team Analysis (2014-2023)")
st.write(
    "Explore the fascinating world of EA Sports FIFA video games with our data analysis app. We've delved into the data \
    spanning FIFA 14 to FIFA 24 to uncover insights into the performance, attributes, and trends of virtual soccer teams.\
    Get to know the best teams in crucial attributes, and find out which leagues are dominating the virtual soccer scene.\
 This analysis provides a snapshot of this virtual soccer universe during this exciting decade.\
Select your areas of interest from the dropdown menus and let the data tell the story. Enjoy your exploration!"
)


# Data Wrangling section
st.subheader("Data overview")
st.write("Let's start by exploring the data and its attributes.")

# Show first rows of the data
if st.checkbox("Show Data"):
    st.dataframe(teams_data.head())

# Number of players in the dataset
if st.checkbox('Show Number of Teams'):
    num_teams = teams_data['team_id'].nunique()
    st.write(f"Number of teams in the dataset: {num_teams}")


# List of columns to convert to numeric
columns_to_convert = [
    'overall', 'attack', 'midfield', 'defence',
    'international_prestige', 'domestic_prestige', 'transfer_budget_eur',
    'club_worth_eur', 'starting_xi_average_age', 'whole_team_average_age',
    'def_style', 'def_team_width', 'def_team_depth', 'def_defence_pressure',
    'def_defence_aggression', 'def_defence_width', 
     'off_team_width', 'off_players_in_box', 'build_up_play_speed',
    'build_up_play_dribbling', 'build_up_play_passing', 'chance_creation_passing',
    'chance_creation_crossing', 'chance_creation_shooting'
]

# Loop through the columns and convert them to numeric
for column in columns_to_convert:
    teams_data[column] = pd.to_numeric(teams_data[column], errors='coerce')

# Descriptive statistics
st.subheader("Descriptive Statistics")
if st.checkbox("Show Descriptive Statistics of Teams Attributes"):
    st.write(teams_data.describe())


# Add a checkbox to show the pie chart
show_pie_chart = st.checkbox("Show Pie Chart of Teams Chance Creation Positioning")

if show_pie_chart:
    category_counts = teams_data['chance_creation_positioning'].value_counts()
    
    # Create a pie chart using Plotly Express
    fig = px.pie(
        values=category_counts,
        names=category_counts.index,
        title='Teams Chance Creation Distribution'
    ) 
    # Display the pie chart
    st.plotly_chart(fig)
else:
    st.write("Check the box above to show distribution of player positions.")

    

# Add a checkbox to show the pie chart
show_build_up_chart = st.checkbox("Show Pie Chart of Teams Build Up Play Positioning")

if show_build_up_chart:
    # Count the number of players in each category
    category_build_up_counts = teams_data['build_up_play_positioning'].value_counts()
    
    # Create a pie chart using Plotly Express
    fig = px.pie(
        values=category_build_up_counts,
        names=category_build_up_counts.index,
        title='Teams Build Up Play Distribution'
    ) 
    # Display the pie chart
    st.plotly_chart(fig)
else:
    st.write("Check the box above to show distribution of Build Up Play by Teams.")

# # Create a checkbox to display the trend

teams_data['update_as_of'] = pd.to_datetime(teams_data['update_as_of'], format='%m/%d/%Y', errors='coerce')

# Extract the year from the 'date' column and create a new column 'year'
teams_data['year'] = teams_data['update_as_of'].dt.year
show_trend = st.checkbox('Show Trend of Average Overall Rating Overtime')

if show_trend:
    # Convert the 'date' column to datetime format
    teams_data['update_as_of'] = pd.to_datetime(teams_data['update_as_of'], format='%m/%d/%Y', errors='coerce')

    # Extract the year from the 'date' column and create a new column 'year'
    teams_data['year'] = teams_data['update_as_of'].dt.year

    # Group the data by 'year' and calculate the average overall rating for each year
    average_ratings_by_year = teams_data.groupby('year')['overall'].mean()

    # Create a line plot to show the trend
    st.write('Trend of Average Overall Rating of Teams Over the Years:')
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(average_ratings_by_year.index, average_ratings_by_year.values, marker='o', color='grey', linestyle='-')
    ax.set_xlabel('Year')
    ax.set_ylabel('Average Overall Rating')
    ax.set_title('Trend of Average Overall Rating')
    
    # Set the y-axis limit to start from 0 and end at 100
    ax.set_ylim(50, 80)
    ax.grid(True)

    # Add data labels to the data points
    for year, rating in zip(average_ratings_by_year.index, average_ratings_by_year.values):
        ax.text(year, rating, f'{rating:.2f}', ha='center', va='bottom')

    st.pyplot(fig)


# # Create a multiselect dropdown to select a team
selected_teams = st.multiselect('Select a Team to Show the Trend of Average Rating Overtime', teams_data['team_name'].unique())

if selected_teams:
    selected_team = selected_teams[0]  # Get the first (and presumably only) selected team

    # Filter data based on the selected team
    team_data = teams_data[teams_data['team_name'] == selected_team]

    # Convert the 'date' column to datetime format
    team_data['update_as_of'] = pd.to_datetime(team_data['update_as_of'])

    # Extract the year from the 'date' column and create a new column 'year'
    team_data['year'] = team_data['update_as_of'].dt.year

    # Group the data by 'year' and calculate the average overall rating for each year
    average_ratings_by_year = team_data.groupby('year')['overall'].mean()

    # Create a line plot to show the trend
    st.write(f'Trend of Average Overall Rating for {selected_team} Over the Years:')
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(average_ratings_by_year.index, average_ratings_by_year.values, marker='o', linestyle='-', color='grey')
    
    # Set the y-axis limit to start from 0
    ax.set_ylim(50, 100)

    ax.set_xlabel('Year')
    ax.set_ylabel('Average Overall Rating')
    ax.set_title(f'Trend of Average Overall Rating for {selected_team}')
    ax.grid(True)

    # Add data labels to the data points
    for year, rating in zip(average_ratings_by_year.index, average_ratings_by_year.values):
        ax.text(year, rating, f'{rating:.2f}', ha='center', va='bottom')

    st.pyplot(fig)




# Create a multiselect dropdown to select a player
selected_leagues = st.multiselect('Select a League', teams_data['league_name'].unique())

if selected_leagues:
    selected_leg = selected_leagues[0]  # Get the first (and presumably only) selected league

    # Filter data based on the selected player
    league_data = teams_data[teams_data['league_name'] == selected_leg]

    # Convert the 'date' column to datetime format
    league_data['update_as_of'] = pd.to_datetime(league_data['update_as_of'])

    # Extract the year from the 'date' column and create a new column 'year'
    league_data['year'] = league_data['update_as_of'].dt.year

    # Group the data by 'year' and calculate the average overall rating for each year
    league_average_ratings_by_year = league_data.groupby('year')['overall'].mean()

    # Create a line plot to show the trend
    st.write(f'Trend of Average Overall Rating for {selected_leg} Over the Years:')
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(league_average_ratings_by_year.index, league_average_ratings_by_year.values, marker='o', linestyle='-', color='grey')
    ax.set_xlabel('Year')
    ax.set_ylabel('Average Overall Rating')
    ax.set_title(f'Trend of Average Overall Rating for {selected_leg}')
    ax.grid(True)

    # Set the y-axis limit to start from 0 and end at 100
    ax.set_ylim(50, 85)

    # Add data labels to the data points
    for year, rating in zip(league_average_ratings_by_year.index, league_average_ratings_by_year.values):
        ax.text(year, rating, f'{rating:.2f}', ha='center', va='bottom')

    st.pyplot(fig)

numerical_data = teams_data.select_dtypes(include=['number'])

# Data Visualization section
st.subheader("Distribution of Teams Attributes")
st.write("Select a team attribute to see their distribution.")

# Select attributes to plot
selected_variable = st.multiselect("Select a team attribute to show distribution", numerical_data.columns[1:])

# Check if any attribute has been selected
if selected_variable:
    # Display histogram
    st.subheader(f"Histogram of {', '.join(selected_variable)}")
    plt.figure(figsize=(10, 6))
    for variable in selected_variable:
        plt.hist(teams_data[variable], bins=20, alpha=0.7, edgecolor='black', label=variable)
    plt.xlabel("Attribute Values")
    plt.ylabel("Frequency")
    plt.title(f"Histogram of {', '.join(selected_variable)}")
    plt.legend()
    st.pyplot()


# Correlation heatmap
corr_columns = teams_data[['overall', 'attack', 'midfield', 'defence', 
       'def_team_width', 'def_team_depth', 'def_defence_pressure',
       'def_defence_aggression', 'def_defence_width', 'off_team_width','off_players_in_box', 
       'build_up_play_speed', 'build_up_play_dribbling',
       'build_up_play_passing', 'chance_creation_passing',
       'chance_creation_crossing', 'chance_creation_shooting']]

st.subheader("Show Correlation Heatmap of Team Attributes")
st.write("Explore the relationships between key teams attributes.")
if st.checkbox("Show Correlation Heatmap"):
    # Create a heatmap of attribute correlations
    plt.figure(figsize=(20, 16))
    sns.heatmap(corr_columns.corr(), annot=True, fmt='.2f', cmap = sns.color_palette("RdBu", 100))
    plt.title("Heatmap Showing Relationship Between Teams Attributes")
    st.pyplot()


# Create a Streamlit app
st.subheader("Top 10 Teams by Attributes")

# Dropdown to select a team attribute
attribute_options = ['overall', 'attack', 'midfield', 'defence', 'international_prestige',
       'domestic_prestige', 'def_team_width', 'def_team_depth', 'def_defence_pressure',
       'def_defence_aggression', 'def_defence_width', 'off_team_width','off_players_in_box', 
       'build_up_play_speed', 'build_up_play_dribbling', 'starting_xi_average_age', 'whole_team_average_age',
       'build_up_play_passing', 'chance_creation_passing',
       'chance_creation_crossing', 'chance_creation_shooting']

selected_attribute = st.selectbox("Select a team attribute", [''] + attribute_options)

if selected_attribute:
    # Calculate the average of the selected attribute for each team
    team_data = teams_data.groupby(['team_name']).agg({
        'league_name': 'last',
        selected_attribute: 'mean'
    }).reset_index()

    # Sort the player data by the average of the selected attribute in descending order
    team_data = team_data.sort_values(by=selected_attribute, ascending=False).head(10)

    # Display the top 10 players with the highest average score in the selected attribute
    st.write(f"Top 10 Teams with Highest {selected_attribute}")
    st.table(team_data)

    if not team_data.empty:
    # Create a bar chart to visualize the top 10 players
        fig, ax = plt.subplots(figsize=(10, 6))
        team_data.plot(x='team_name', y=selected_attribute, kind='bar', ax=ax, color='skyblue')
        plt.xlabel("Team Name")
        plt.ylabel(selected_attribute)
        plt.title(f"Top 10 Teams with Highest {selected_attribute}")
    
    # Add data labels to the bars with 2 decimal places
    for i, v in enumerate(team_data[selected_attribute][:10]):
        ax.text(i, v, f"{v:.2f}", ha='center', va='bottom')

    st.pyplot(fig)
else:
    st.warning("Please select a team attribute to view the top teams.")


# st.write("<h2>Top 10 Teams by Average Overall Rating of Players</h2>", unsafe_allow_html=True)
st.write("Top Teams by Average Overall Rating Over Time")

# Group by player_name and calculate the average overall rating
top_10_teams = teams_data.groupby('team_name')[['overall']].mean().sort_values(by='overall', ascending=False).head(10)

# Plot the top players
if st.checkbox("Show Top 10 Teams"):
    plt.figure(figsize=(12, 6))
    bars = plt.barh(top_10_teams.index, top_10_teams['overall'], color='skyblue')
    plt.xlabel('Average Overall Rating')
    
    st.write('Top 10 Teams by Average Overall Rating Over Time')
    plt.gca().invert_yaxis()  # Invert the y-axis for better visualization

    # Add data labels to the bars
    for bar in bars:
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height() / 2, f'{width:.2f}', ha='left', va='center', color='black')

    st.pyplot()


# Top leagues by Average Overall Rating section
st.write("Top 10 Leagues by Average Overall Rating of Teams")

# Group by player_name and calculate the average overall rating
top_10_leagues = teams_data.groupby('league_name')[['overall']].mean().sort_values(by='overall', ascending=False).head(10)

# Plot the top players
if st.checkbox("Show Top 10 Leagues"):
    plt.figure(figsize=(12, 6))
    bars = plt.barh(top_10_leagues.index, top_10_leagues['overall'], color='skyblue')
    plt.xlabel('Average Overall Rating by League')
    plt.title('Top 10 Leagues by Average Overall Rating of their Teams')
    plt.gca().invert_yaxis()  # Invert the y-axis for better visualization

    # Add data labels to the bars
    for bar in bars:
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height() / 2, f'{width:.2f}', ha='left', va='center', color='black')

    st.pyplot()



st.write("Select Option to Find Teams with the Youngest Squad")

# Group by player_name and calculate the average overall rating
youngest_teams = teams_data.groupby('team_name')[['whole_team_average_age']].mean().sort_values(by='whole_team_average_age', ascending=True).head(10)

# Plot the top players
if st.checkbox("Show Top 10 Teams with the Youngest Squad"):
    plt.figure(figsize=(12, 6))
    bars = plt.barh(youngest_teams.index, youngest_teams['whole_team_average_age'], color='skyblue')
    plt.xlabel('Average Age by Team')
    plt.title('Top 10 Leagues by Average Age of their Players')
    plt.gca().invert_yaxis()  # Invert the y-axis for better visualization

    # Add data labels to the bars
    for bar in bars:
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height() / 2, f'{width:.2f}', ha='left', va='center', color='black')

    st.pyplot()


st.write("Select Option to Find Teams with Oldest Squad")

# Group by player_name and calculate the average overall rating
oldest_teams = teams_data.groupby('team_name')[['whole_team_average_age']].mean().sort_values(by='whole_team_average_age', ascending=False).head(10)

# Plot the top players
if st.checkbox("Show Top 10 Teams with Oldest Squad"):
    plt.figure(figsize=(12, 6))
    bars = plt.barh(oldest_teams.index, oldest_teams['whole_team_average_age'], color='skyblue')
    plt.xlabel('Average Age by Team')
    plt.title('Top 10 Teams by Average Age of their Players')
    plt.gca().invert_yaxis()  # Invert the y-axis for better visualization

    # Add data labels to the bars
    for bar in bars:
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height() / 2, f'{width:.2f}', ha='left', va='center', color='black')

    st.pyplot()


st.write("Select Option to Find Leagues with Oldest Squad")

# Group by player_name and calculate the average overall rating
oldest_leagues = teams_data.groupby('league_name')[['whole_team_average_age']].mean().sort_values(by='whole_team_average_age', ascending=False).head(10)

# Plot the top players
if st.checkbox("Show Top 10 Leagues with the Oldest squad"):
    plt.figure(figsize=(12, 6))
    bars = plt.barh(oldest_leagues.index, oldest_leagues['whole_team_average_age'], color='skyblue')
    plt.xlabel('Average Age by Team')
    plt.title('Top 10 Leagues by Average Age of their Players')
    plt.gca().invert_yaxis()  # Invert the y-axis for better visualization

    # Add data labels to the bars
    for bar in bars:
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height() / 2, f'{width:.2f}', ha='left', va='center', color='black')

    st.pyplot()


st.write("Select from the dropdown options to find Leagues with youngest squad")

# Group by player_name and calculate the average overall rating
youngest_leagues = teams_data.groupby('league_name')[['whole_team_average_age']].mean().sort_values(by='whole_team_average_age', ascending=True).head(10)

# Plot the top players
if st.checkbox("Show Top 10 Leagues with the Youngest squad"):
    plt.figure(figsize=(12, 6))
    bars = plt.barh(youngest_leagues.index, youngest_leagues['whole_team_average_age'], color='skyblue')
    plt.xlabel('Average Age by Team')
    plt.title('Top 10 Leagues by Average Age of their Players')
    plt.gca().invert_yaxis()  # Invert the y-axis for better visualization

    # Add data labels to the bars
    for bar in bars:
        width = bar.get_width()
        plt.text(width, bar.get_y() + bar.get_height() / 2, f'{width:.2f}', ha='left', va='center', color='black')

    st.pyplot()


# Create a dropdown to select the variable
selected_variable = st.selectbox("Select a Variable:", [
    'attack', 'midfield', 'defence', 'international_prestige',
    'domestic_prestige', 'def_team_width', 'def_team_depth',
    'def_defence_pressure', 'def_defence_aggression',
    'def_defence_width', 'off_team_width', 'off_players_in_box',
    'build_up_play_speed', 'build_up_play_dribbling',
    'starting_xi_average_age', 'whole_team_average_age',
    'build_up_play_passing', 'chance_creation_passing',
    'chance_creation_crossing', 'chance_creation_shooting'
])

if selected_variable:
    st.write(f"Overall Rating vs. {selected_variable}")

    # Group by the selected variable and calculate the mean of 'Overall' rating
    variable_rating_mean = teams_data.groupby(selected_variable)['overall'].mean().reset_index()

    # Plot the data
    plt.figure(figsize=(10, 6))
    plt.plot(variable_rating_mean[selected_variable], variable_rating_mean['overall'], color='grey')
    plt.title(f"{selected_variable} vs. Overall Rating", fontsize=16, fontweight='bold')
    plt.xlabel(selected_variable, fontsize=12)
    plt.ylabel("Overall Rating", fontsize=12)
    plt.grid(color='grey', linestyle='--', linewidth=0.5)

    st.pyplot(plt)


# Display your name in the sidebar
st.sidebar.write("Developed by Simon-Peter Osadiapét")

# Display your picture next to your name
picture = "osadiapet.jpg"  
st.sidebar.image(picture, use_column_width=True, caption="Developed by Simon-Peter Osadiapét")