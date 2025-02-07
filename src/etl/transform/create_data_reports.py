
import pandas as pd
from tabulate import tabulate
import os

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
        # extract the value from the column


        #if not pd.isna(value):
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


def create_report(dfs, output_dir):
    for name, df in dfs.items():
        data_gaps_dict = {}
        output_lines = []
        output_lines.append(f"{name}\n")

        metrics = ["Number of data points", "Median gap duration", "Number of gaps between 100 and 140 minutes", "Number of gaps >140 minutes", "Gaps > 100 minutes", "First data point date"]
        results = {metric: [] for metric in metrics}
        results["Metric"] = metrics

        # loop through all columns except timestamp and datetime
        cols_to_loop = [col for col in df.columns if col not in ['timestamp_tree', 'datetime']]
        
        first_dates = {}

        for col in cols_to_loop:
            df_tmp = df[['timestamp_tree', 'datetime', col]].copy().dropna()
            gaps_df = calculate_data_gaps(df_tmp, column=col)
            data_gaps_dict[name] = gaps_df
            
            num_data_points = gaps_df.shape[0]
            median_gap_duration = gaps_df['Duration'].median()
            count_100_140 = gaps_df[(gaps_df['Duration'] >= pd.Timedelta(minutes=100)) & (gaps_df['Duration'] <= pd.Timedelta(minutes=140))].shape[0]
            gaps_over_100 = gaps_df[gaps_df['Duration'] > pd.Timedelta(minutes=100)][['Start Time', 'Duration']]
            gaps_over_140 = gaps_df[gaps_df['Duration'] > pd.Timedelta(minutes=140)][['Start Time', 'Duration']]
            num_gaps_over_140 = gaps_over_140.shape[0]
            top_5_gaps = gaps_over_140.nlargest(5, 'Duration')
            first_date = df_tmp['datetime'].iloc[0]

            first_dates[col] = first_date

            results["Number of data points"].append(num_data_points)
            results["Median gap duration"].append(median_gap_duration)
            results["Number of gaps between 100 and 140 minutes"].append(count_100_140)
            results["Number of gaps >140 minutes"].append(num_gaps_over_140)
            results["First data point date"].append(first_date)
            if not gaps_over_100.empty:
                gaps_txt = "\n".join([f"{row['Start Time']} - {row['Duration']}" for _, row in gaps_over_100.iterrows()])
            else:
                gaps_txt = "No gaps > 140 minutes"
            results["Gaps > 100 minutes"].append(gaps_txt)
        # Create a list of lists to represent the table
        table_data = []
        for metric in metrics:
            row = [metric] + results[metric]
            table_data.append(row)

        # Print the summary table
        headers = ["Metric"] + cols_to_loop
        table = tabulate(table_data, headers=headers, tablefmt='grid')
        output_lines.append(table)
        output_lines.append("\n" + "-"*50 + "\n\n")

        '''
        # Calculate and print the difference in first dates
        if "-10|ENV__SOIL__VWC" in first_dates and "TOP|ENV__ATMO__T_y" in first_dates:
            date_diff = first_dates["-10|ENV__SOIL__VWC"] - first_dates["TOP|ENV__ATMO__T_y"]
            date_diff_str = f"Time btw. first row of weather vs tree sensor: {date_diff}"
            output_lines.append(date_diff_str)
        output_lines.append("\n")
        '''
        # Write the output to a text file
        with open(os.path.join(output_dir, f"{name}.txt"), "w") as f:
            f.write("\n".join(output_lines))


if __name__ == "__main__":
    print("Calculating data gaps...\n")
    # Load the data data\merged\Baum Brucker See_+_Wetter Bruck_merged_v2.csv
    abs_path  = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "merged"))
    output_abs_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "data", "merged", "quality_reports"))
    #get all csv files in the directory
    files = os.listdir(abs_path)
    # loop through all files
    for file in files:
        if file.endswith(".csv"):
            # get filename without extension
            filename = os.path.splitext(file)[0]
            input_abs_path = os.path.join(abs_path, file)
            df = pd.read_csv(input_abs_path)
            create_report({filename: df}, output_abs_path)

