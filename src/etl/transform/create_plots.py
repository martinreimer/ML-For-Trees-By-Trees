import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns
import os


plt.rcParams.update({
    'font.size': 34,  # Global font size
    'axes.titlesize': 34,  # Title font size
    'axes.labelsize': 34,  # X and Y label size
    'xtick.labelsize': 34,  # X-axis tick label size
    'ytick.labelsize': 34,  # Y-axis tick label size
    'legend.fontsize': 34,  # Legend font size
    'figure.titlesize': 34  # Figure title size
})
sns.set_theme()


# Check Data Gaps

def calculate_data_gaps(df, column):
    """
    Calculate the data gaps for the given dataframe and column.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing the data
    column (str): Column name for which to calculate the gaps
    
    Returns:
    pd.DataFrame: DataFrame with the start time, end time, and duration of each gap
    """
    df = df.sort_values(by='datetime')
    df['datetime'] = pd.to_datetime(df['datetime'])
    
    start_time = None
    prev_valid_time = None
    results = []

    for index, row in df.iterrows():
        current_time = row['datetime']
        value = row[column]

        if not pd.isna(value):
            if prev_valid_time is not None:
                duration = current_time - prev_valid_time
                results.append({
                    'Start Time': prev_valid_time,
                    'End Time': current_time,
                    'Duration': duration
                })
            prev_valid_time = current_time

    # If the first value is NaN, find the first valid value
    if prev_valid_time is None:
        for index, row in df.iterrows():
            value = row[column]
            if not pd.isna(value):
                start_time = row['datetime']
                results.append({
                    'Start Time': df.iloc[0]['datetime'],
                    'End Time': start_time,
                    'Duration': start_time - df.iloc[0]['datetime']
                })
                break

    return pd.DataFrame(results)

def plot_column_with_gaps(df, gaps_df, column, title, gap_threshold_minutes=140, show_gap_text=False, save_img=False, img_path=None):   
    """
    Plot the VWC values and highlight the data gaps.
    """
    
    # Primary y-axis for soil moisture (VWC) data
    fig, ax1 = plt.subplots(figsize=(12, 8))

    #ax1.set_facecolor('white')
    ax1.grid(False)
    ax1.plot(df['datetime'], df[column], label=column, color='black')
    ax1.set_xlabel('Date', fontsize=18)
    ax1.set_ylabel(column, color='black', fontsize=18)
    ax1.tick_params(axis='y', labelcolor='black', labelsize=18)
    plt.xticks(rotation=45, fontsize=18)

    # Highlight data gaps
    
    for _, interval in gaps_df.iterrows():
        if interval['Duration'] > pd.Timedelta(minutes=gap_threshold_minutes):
            ax1.axvspan(interval['Start Time'], interval['End Time'], color='red', alpha=0.1)
            if show_gap_text:
                ax1.text(interval['Start Time'], 0.5, f'{interval["Duration"]} ({interval["Start Time"].date()})', 
                         rotation=90, verticalalignment='center')
    
    # Plot irrigation events

    # Legends
    legend_elements = [Line2D([0], [0], color='red', lw=3, label='Data Gap'),
                       Line2D([0], [0], color='black', lw=1.5, label=column),
                       ]
    
    ax1.legend(handles=legend_elements, loc='upper left', fontsize=18)
    plt.tight_layout()
    # Save the image if needed
    plt.savefig(img_path)


def plot_column_with_barplot(df, gaps_df, column, barplot_column, title, gap_threshold_minutes=140, show_gap_text=False, save_img=False, img_path=None):   
    """
    Plot the VWC values and highlight the data gaps.
    Additionally, plot the rain_delta data on a secondary y-axis with adjustable limits.
    
    """
    plt.figure(figsize=(16, 10))
    
    # Primary y-axis for soil moisture (VWC) data
    fig, ax1 = plt.subplots(figsize=(16, 10))
    #ax1.set_facecolor('white')
    ax1.grid(False)
    ax1.plot(df['datetime'], df[column], label=column, color='black')
    ax1.set_xlabel('Date')
    ax1.set_ylabel(column, color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    plt.xticks(rotation=45)

    # Highlight data gaps
    for _, interval in gaps_df.iterrows():
        if interval['Duration'] > pd.Timedelta(minutes=gap_threshold_minutes):
            ax1.axvspan(interval['Start Time'], interval['End Time'], color='red', alpha=0.1)
            if show_gap_text:
                ax1.text(interval['Start Time'], 0.5, f'{interval["Duration"]} ({interval["Start Time"].date()})', 
                         rotation=90, verticalalignment='center')

    # Create a second axis for the bar plot 
    # Extract month and year for aggregation
    df['year_month'] = df['datetime'].dt.to_period('M')
    # Aggregate precipitation values per month
    monthly_precipitation = df.groupby('year_month')[barplot_column].sum()
    # Calculate the start and end of each month for stretching the bars
    months = monthly_precipitation.index.to_timestamp()  # Convert PeriodIndex to Timestamp for plotting
    month_starts = months.to_series().apply(lambda x: x.replace(day=1))
    # Create a second axis for the bar plot (precipitation)
    ax2 = ax1.twinx()
    # Plot bars from the start to end of each month, with the corresponding width
    ax2.bar(month_starts, monthly_precipitation, width=30, align='edge', 
            color='blue', alpha=0.6, label='Monthly Precipitation')

    ax2.set_ylabel('Agg. Monthly Precipitation (mm)', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')
    ax2.set_ylim(0, 400)  # Adjust this as needed

    # Legends
    legend_elements = [Line2D([0], [0], color='red', lw=3, label='Data Gap'),
                       Line2D([0], [0], color='black', lw=1.5, label=column),
                       Line2D([0], [0], color='blue', lw=1.5, label='Rain Delta')]
    
    ax1.legend(handles=legend_elements, loc='upper left')
    # Save the image if needed
    plt.savefig(img_path)

def plot_column_with_gaps_irrigations_rain(df, gaps_df, column, irrigation_column, rain_column, title, gap_threshold_minutes=140, show_gap_text=False, save_img=False, img_path=None):   
    """
    Plot the VWC values and highlight the data gaps.
    Additionally, plot the rain_delta data on a secondary y-axis with adjustable limits.
    
    Parameters:
    df (pd.DataFrame): DataFrame containing the data
    gaps_df (pd.DataFrame): DataFrame containing the gaps
    column (str): Column name for which to plot the data (e.g., soil moisture)
    irrigation_column (str): Column name for irrigation events
    title (str): Title for the plot
    gap_threshold_minutes (int): Minimum gap duration to be highlighted (in minutes)
    show_gap_text (bool): Whether to show gap duration text on the plot
    save_img (bool): Whether to save the plot as an image
    img_name (str): Name of the image file to be saved
    """
    plt.figure(figsize=(12, 6))
    
    # Primary y-axis for soil moisture (VWC) data
    fig, ax1 = plt.subplots(figsize=(12, 6))
    #ax1.set_facecolor('white')
    ax1.grid(False)
    ax1.plot(df['datetime'], df[column], label=column, color='black')
    ax1.set_xlabel('Date')
    ax1.set_ylabel(column, color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    plt.xticks(rotation=45)

    # Highlight data gaps
    for _, interval in gaps_df.iterrows():
        if interval['Duration'] > pd.Timedelta(minutes=gap_threshold_minutes):
            ax1.axvspan(interval['Start Time'], interval['End Time'], color='red', alpha=0.1)
            if show_gap_text:
                ax1.text(interval['Start Time'], 0.5, f'{interval["Duration"]} ({interval["Start Time"].date()})', 
                         rotation=90, verticalalignment='center')

    # Plot irrigation events
    irrigation_events = df[df[irrigation_column].notnull() & df[irrigation_column] > 0][['datetime', irrigation_column]]
    for index, row in irrigation_events.iterrows():
        irrigation_date = row['datetime']
        ax1.axvline(x=irrigation_date, color='green', linewidth=1)

    # Secondary y-axis for rain_delta data
    ax2 = ax1.twinx()
    ax2.grid(False)
    ax2.bar(df['datetime'], df[rain_column], width=1, align='edge', 
            color='blue', alpha=0.6, label='Monthly Precipitation')
    #ax2.plot(df['datetime'], df[rain_column], label='Rain Delta', color='blue')
    ax2.set_ylabel('Rain Delta', color='blue')
    ax2.tick_params(axis='y', labelcolor='blue')

    # Set the limit for rain_delta y-axis
    ax2.set_ylim(0, 50)  # Adjust this as needed
    
    # Legends
    legend_elements = [Line2D([0], [0], color='red', lw=3, label='Data Gap'),
                       Line2D([0], [0], color='green', lw=3, label='Irrigation Event'),
                       Line2D([0], [0], color='black', lw=1.5, label=column),
                       Line2D([0], [0], color='blue', lw=1.5, label='Rain Delta')]
    
    ax1.legend(handles=legend_elements, loc='upper left')

    # Save the image if needed
    plt.savefig(img_path) 


if __name__ == "__main__":
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    # create plots for tree data
    print("Creating plots for tree data")
    input_abs_path  = os.path.abspath(os.path.join(cur_dir, "..", "..", "..", "data", "raw", "climavi", "tree_sensors"))
    output_abs_path = os.path.abspath(os.path.join(cur_dir, "..", "..", "..", "media", "trees"))
    files = os.listdir(input_abs_path)
    col_name = '-10|ENV__SOIL__VWC'
    dfs = {}
    for file in files:
        if file.endswith(".csv"):
            # get filename without extension
            filename = os.path.splitext(file)[0]
            csv_path = os.path.join(input_abs_path, file)
            df = pd.read_csv(csv_path)
            # create daily df, we only need the daily minimum value of the column '-10|ENV__SOIL__VWC'
            df = df[['timestamp', 'datetime', col_name]]
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            #resample per column
            df = df.dropna(subset=[col_name])
            daily_min = df.loc[df.groupby(df['datetime'].dt.date)[col_name].idxmin()]
            daily_min = daily_min.reset_index(drop=True)
            dfs[filename] = daily_min
    for name, df in dfs.items():
        #df['datetime'] = pd.to_datetime(df['datetime'])
        print(f"{name}")
        gaps_df = calculate_data_gaps(df, column=col_name)
        # Plot the data gaps
        img_path = os.path.join(output_abs_path, f"{name}.jpg")
        plot_column_with_gaps(df=df, gaps_df=gaps_df, column=col_name, title=f'{name}', gap_threshold_minutes=60*48, show_gap_text=False, save_img=True, img_path=img_path)

    # create plots for climavi weather data
    print("Creating plots for weather data")
    input_abs_path  = os.path.abspath(os.path.join(cur_dir, "..", "..", "..", "data", "raw", "climavi", "weather_sensors"))
    output_abs_path = os.path.abspath(os.path.join(cur_dir, "..", "..", "..", "media", "climavi_weather"))
    files = os.listdir(input_abs_path)
    col_name = 'TOP|ENV__ATMO__T'
    rain_col = 'EXT|ENV__ATMO__RAIN__DELTA'
    dfs = {}
    for file in files:
        if file.endswith(".csv"):
            # get filename without extension
            filename = os.path.splitext(file)[0]
            csv_path = os.path.join(input_abs_path, file)
            df = pd.read_csv(csv_path)
            # create daily df, we only need the daily minimum value of the column '-10|ENV__SOIL__VWC'
            df = df[['timestamp', 'datetime', col_name, rain_col]]
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            #resample per column
            df = df.dropna(subset=[col_name, rain_col])
            # Aggregate to get the daily minimum temperature and total daily precipitation
            df['date'] = df['datetime'].dt.date
            daily_data = df.groupby('date').agg({
                col_name: 'min',     # Daily minimum temperature
                rain_col: 'sum',    # Daily total precipitation,
                "datetime": "first"
            }).reset_index()
            dfs[filename] = daily_data
    for name, df in dfs.items():
        #df['datetime'] = pd.to_datetime(df['datetime'])
        print(f"{name}")
        gaps_df = calculate_data_gaps(df, column=col_name)
        # Plot the data gaps
        img_path = os.path.join(output_abs_path, f"{name}.jpg")
        plot_column_with_barplot(df=df, gaps_df=gaps_df, column=col_name, barplot_column=rain_col, title="Temperature and monthly agg. rain and data gap > 1 day", gap_threshold_minutes=60*72, show_gap_text=False, save_img=False, img_path=img_path)

    # create plots for openmeteo weather hourly data
    print("Creating plots for weather data")
    input_abs_path  = os.path.abspath(os.path.join(cur_dir, "..", "..", "..", "data", "raw", "openmeteo", "Weather_Hourly.csv"))
    output_abs_path = os.path.abspath(os.path.join(cur_dir, "..", "..", "..", "media", "openmeteo_weather"))
    col_name = 'temperature_2m'
    rain_col = 'precipitation'
    dfs = {}

    df = pd.read_csv(input_abs_path)
    # create daily df, we only need the daily minimum value of the column '-10|ENV__SOIL__VWC'
    df = df[['date', col_name, rain_col]]
    df['datetime'] = pd.to_datetime(df['date'])
    #resample per column
    df = df.dropna(subset=[col_name, rain_col])
    # Aggregate to get the daily minimum temperature and total daily precipitation
    df['date'] = df['datetime'].dt.date
    daily_data = df.groupby('date').agg({
        col_name: 'min',     # Daily minimum temperature
        rain_col: 'sum',    # Daily total precipitation,
        "datetime": "first"
    }).reset_index()
    dfs[filename] = daily_data
    for name, df in dfs.items():
        #df['datetime'] = pd.to_datetime(df['datetime'])
        print(f"{name}")
        gaps_df = calculate_data_gaps(df, column=col_name)
        # Plot the data gaps
        img_path = os.path.join(output_abs_path, f"openmeteo.jpg")
        plot_column_with_barplot(df=df, gaps_df=gaps_df, column=col_name, barplot_column=rain_col, title="Temperature and monthly agg. rain and data gap > 1 day", gap_threshold_minutes=60*72, show_gap_text=False, save_img=False, img_path=img_path)



    # create plots for merged climavi data
    print("Creating plots for merged data")
    input_abs_path  = os.path.abspath(os.path.join(cur_dir, "..", "..", "..", "data", "merged", "climavi_tree_climavi_weather", "final"))
    output_abs_path = os.path.abspath(os.path.join(cur_dir, "..", "..", "..", "media", "merged_climavi"))
    files = os.listdir(input_abs_path)
    col_name = '-10|ENV__SOIL__VWC' # Column name to calculate data gaps for
    irrigation_column = "DOC|ENV__SOIL__IRRIGATION"
    rain_column = "EXT|ENV__ATMO__RAIN__DELTA"
    dfs = {}
    for file in files:
        if file.endswith("_merged_processed.csv"):
            # get filename without extension
            filename = os.path.splitext(file)[0]
            csv_path = os.path.join(input_abs_path, file)
            df = pd.read_csv(csv_path)

            # create daily df, we only need the daily minimum value of the column '-10|ENV__SOIL__VWC'
            df = df[['timestamp_tree', 'datetime', col_name, rain_column, irrigation_column]]
            df['datetime'] = pd.to_datetime(df['timestamp_tree'], unit='ms')
            #resample per column
            df = df.dropna(subset=[col_name, rain_column])
            # Aggregate to get the daily minimum temperature and total daily precipitation
            df['date'] = df['datetime'].dt.date
            daily_data = df.groupby('date').agg({
                col_name: 'min',     # Daily minimum temperature
                rain_column: 'sum',    # Daily total precipitation,
                "datetime": "first",
                "timestamp_tree": "first",
                irrigation_column: 'max'
            }).reset_index()
            dfs[filename] = daily_data

    
    for name, df in dfs.items():
        df['datetime'] = pd.to_datetime(df['datetime'])
        print(f"{name}")
        gaps_df = calculate_data_gaps(df, column=col_name)
        
        # Plot the data gaps
        img_path = os.path.join(output_abs_path, f"{name}.jpg")
        plot_column_with_gaps_irrigations_rain(df=df, gaps_df=gaps_df, column=col_name, irrigation_column=irrigation_column, rain_column=rain_column, title=f'{name}', gap_threshold_minutes=60*48, show_gap_text=False, save_img=True, img_path=img_path)


    # create plots for merged openmeteo data
    print("Creating plots for merged openmeteo data")
    input_abs_path  = os.path.abspath(os.path.join(cur_dir, "..", "..", "..", "data", "merged", "climavi_tree_openmeteo_weather", "final"))
    output_abs_path = os.path.abspath(os.path.join(cur_dir, "..", "..", "..", "media", "merged_climavi_openmeteo"))
    files = os.listdir(input_abs_path)
    col_name = '-10|ENV__SOIL__VWC' # Column name to calculate data gaps for
    irrigation_column = "DOC|ENV__SOIL__IRRIGATION"
    rain_column = "precipitation"
    dfs = {}
    for file in files:
        if file.endswith("_merged_processed.csv"):
            # get filename without extension
            filename = os.path.splitext(file)[0]
            csv_path = os.path.join(input_abs_path, file)
            df = pd.read_csv(csv_path)

            # create daily df, we only need the daily minimum value of the column '-10|ENV__SOIL__VWC'
            df = df[['timestamp', 'datetime', col_name, rain_column, irrigation_column]]
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')
            #resample per column
            df = df.dropna(subset=[col_name, rain_column])
            # Aggregate to get the daily minimum temperature and total daily precipitation
            df['date'] = df['datetime'].dt.date
            daily_data = df.groupby('date').agg({
                col_name: 'min',     # Daily minimum temperature
                rain_column: 'sum',    # Daily total precipitation,
                "datetime": "first",
                "timestamp": "first",
                irrigation_column: 'max'
            }).reset_index()
            dfs[filename] = daily_data

    
    for name, df in dfs.items():
        df['datetime'] = pd.to_datetime(df['datetime'])
        print(f"{name}")
        gaps_df = calculate_data_gaps(df, column=col_name)
        
        # Plot the data gaps
        img_path = os.path.join(output_abs_path, f"{name}.jpg")
        plot_column_with_gaps_irrigations_rain(df=df, gaps_df=gaps_df, column=col_name, irrigation_column=irrigation_column, rain_column=rain_column, title=f'{name}', gap_threshold_minutes=60*48, show_gap_text=False, save_img=True, img_path=img_path)