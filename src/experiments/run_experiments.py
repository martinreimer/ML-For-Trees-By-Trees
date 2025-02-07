import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, root_mean_squared_error, mean_absolute_percentage_error
import plotly.graph_objs as go
import warnings
from sklearn.model_selection import TimeSeriesSplit
import matplotlib.pyplot as plt
from datetime import datetime
import pickle
import json
from get_engineered_features import load_daily_data, load_feature_groups, load_hourly_data
warnings.filterwarnings("ignore")



def interactive_plot():
    '''
    Create an interactive plot using Plotly to visualize the daily soil moisture and rain delta data.
    '''
    # Start plotting
    fig = go.Figure()

    # Add daily soil moisture data
    fig.add_trace(
        go.Scatter(x=df_daily.index, y=df_daily['-10|ENV__SOIL__VWC|min'],  # Adjusted column name
                    mode='lines+markers', name='Soil Moisture Min', yaxis='y1', line=dict(color='brown'))
    )

    '''
    # Add daily rain delta data
    fig.add_trace(
        go.Scatter(x=df_daily.index, y=df_daily['precipitation|sum'],  # Adjusted column name
                    mode='lines+markers', name='Rain Delta', yaxis='y2', line=dict(color='blue'))
    )
    '''
    # Add daily rain delta data as a bar plot (histogram-like)
    fig.add_trace(
        go.Bar(x=df_daily.index, y=df_daily['precipitation|sum'],  # Adjusted column name
                name='Rain Delta', yaxis='y2', marker=dict(color='blue'))
    )


    # Add vertical line for irrigation event prediction (event if value is > 0)
    for index, row in df_daily.iterrows():
        if row['DOC_IRRIGATION_PREDICTION|max'] > 0:  # Adjusted column name and condition
            fig.add_shape(
                dict(
                    type="line",
                    x0=row.name,
                    y0=-3,
                    x1=row.name,
                    y1=40,
                    line=dict(
                        color="purple",
                        width=3
                    )
                )
            )

    # Highlight date range from 2024-06-24 to 2024-08-17 with a different background color
    fig.add_shape(
        type="rect",
        x0="2024-06-24", x1="2024-08-29",  # Define the start and end date for the rectangle
        y0=0, y1=1,  # Use y0=0 and y1=1 to cover the entire y-axis height
        xref="x", yref="paper",  # Reference x to the x-axis and y to the paper coordinate
        fillcolor="steelblue",  # Background color
        opacity=0.3,  # Adjust the opacity
        layer="below",  # Place the rectangle behind the plot data
        line_width=0  # No border for the rectangle
    )
    # Add an invisible scatter plot to create a legend entry for the background color
    fig.add_trace(
        go.Scatter(
            x=[None], y=[None],  # Dummy data points for the legend entry
            mode='markers',
            marker=dict(size=10, color='lightgrey', opacity=0.5),  # Match the rectangle color and opacity
            showlegend=True,
            name='Train Data: <2024-06-24'  # Legend text
        )
    )
    # Add an invisible scatter plot to create a legend entry for the background color
    fig.add_trace(
        go.Scatter(
            x=[None], y=[None],  # Dummy data points for the legend entry
            mode='markers',
            marker=dict(size=10, color='steelblue', opacity=0.5),  # Match the rectangle color and opacity
            showlegend=True,
            name='Test Data: 06-24 -> '  # Legend text
        )
    )


    # Update layout for two y-axes
    fig.update_layout(
        title='Soil Moisture and Rain Delta per Day ',
        xaxis_title='Time',
        yaxis=dict(
            title='Soil Moisture',
            titlefont=dict(color='black'),
            tickfont=dict(color='black')
        ),
        yaxis2=dict(
            title='Rain Delta',
            titlefont=dict(color='blue'),
            tickfont=dict(color='blue'),
            overlaying='y',
            side='right',
            range=[0, 50],  # Adjust this range to fit the scale you want on the lower end of the graph
            showgrid=False  # Optionally hide gridlines for secondary y-axis
        )
    )

    # Show the plot
    fig.show()

def get_features_from_groups(df, feature_groups):
    '''
    Get a list of features from the feature groups dictionary that are present in the dataset.
    '''
    features = []
    for group, group_features in feature_groups.items():
        for feature in group_features:
            if feature in df.columns:
                features.append(feature)
            else:
                print(f"Feature {feature} not found in the dataset")
    return features

def split_dataset(df, train_start='2023-05-21', val_start='2024-05-01', test_start='2024-07-15', days_ahead=1, path=None):
    '''
    Split the dataset into train, validation, and test sets based on the specified dates.
    '''
    # Offset to predict n days ahead without data leakage
    date_offset = days_ahead

    # Split into train, train_val, validation, and test sets
    df_train_val = df[(df.index >= train_start) & (df.index < pd.to_datetime(test_start) - pd.DateOffset(days=date_offset))]
    df_train = df[(df.index >= train_start) & (df.index < pd.to_datetime(val_start) - pd.DateOffset(days=date_offset))]
    df_val = df[(df.index >= val_start) & (df.index <= pd.to_datetime(test_start) - pd.DateOffset(days=date_offset))]
    df_test = df[(df.index >= test_start)]

    # print date ranges and len of splits
    print("\nSplitting the dataset into train, validation, and test sets...")
    print(f"Total Rows: {len(df_train_val) + len(df_test)}")
    print(f"Train-Val data: {df_train_val.index.min().date()} - {df_train_val.index.max().date()} ({len(df_train_val)} rows ~ {len(df_train_val) / (len(df_train_val) + len(df_test)) * 100:.0f}% of Total)")
    print(f"   - Train data: {df_train.index.min().date()} - {df_train.index.max().date()} ({len(df_train)} rows ~ {len(df_train) / (len(df_train) + len(df_val) + len(df_test)) * 100:.0f}% of Total - {len(df_train) / (len(df_train) + len(df_val)) * 100:.0f}% of Train-Val)")
    print(f"   - Validation data: {df_val.index.min().date()} - {df_val.index.max().date()} ({len(df_val)} rows ~ {len(df_val) / len(df) * 100:.0f}% of Total - {len(df_val) / (len(df_train) + len(df_val)) * 100:.0f}% of Train-Val)")
    print(f"Test data: {df_test.index.min().date()} - {df_test.index.max().date()} ({len(df_test)} rows ~ {len(df_test) / (len(df_train_val) + len(df_test)) * 100:.0f}%)")
    # also write as txt file
    if path:
        with open(os.path.join(path, "dataset_split.txt"), "w") as f:
            f.write(f"Total Rows: {len(df_train_val) + len(df_test)}\n")
            f.write(f"Train-Val data: {df_train_val.index.min().date()} - {df_train_val.index.max().date()} ({len(df_train_val)} rows ~ {len(df_train_val) / (len(df_train_val) + len(df_test)) * 100:.0f}% of Total)\n")
            f.write(f"   - Train data: {df_train.index.min().date()} - {df_train.index.max().date()} ({len(df_train)} rows ~ {len(df_train) / (len(df_train) + len(df_val) + len(df_test)) * 100:.0f}% of Total - {len(df_train) / (len(df_train) + len(df_val)) * 100:.0f}% of Train-Val)\n")
            f.write(f"   - Validation data: {df_val.index.min().date()} - {df_val.index.max().date()} ({len(df_val)} rows ~ {len(df_val) / len(df) * 100:.0f}% of Total - {len(df_val) / (len(df_train) + len(df_val)) * 100:.0f}% of Train-Val)\n")
            f.write(f"Test data: {df_test.index.min().date()} - {df_test.index.max().date()} ({len(df_test)} rows ~ {len(df_test) / (len(df_train_val) + len(df_test)) * 100:.0f}%)\n")
    return df_train_val, df_train, df_val, df_test

def expanding_window_cv(df_train_val, model, x_features, y_targets, error_metric, tscv):
    """
    Performs expanding window cross-validation.
    
    Parameters:
    df_train_val (pd.DataFrame): The dataframe containing training and validation data.
    model: The machine learning model to evaluate.
    initial_train_size (int): The initial size of the training window.
    val_size (int): The number of samples to use for validation in each iteration.
    step_size (int): The number of steps to expand the window by in each iteration.
    
    Returns:
    errors (list): The list of validation errors (MSE in this case) for each expanding window.
    """

    errors = []
    models = []
    
    for train_index, test_index in tscv.split(df_train_val):
        train_data = df_train_val.iloc[train_index]
        val_data = df_train_val.iloc[test_index]
    
        X_train, y_train = train_data[x_features], train_data[y_targets]
        X_val, y_val = val_data[x_features], val_data[y_targets]
        if len(y_targets) == 1: 
            y_train_values = y_train.values.ravel()
            y_val_values = y_val.values.ravel()
        else:
            y_train_values = y_train.values  
            y_val_values = y_val.values

        # Fit the model
        model.fit(X_train, y_train_values)
        
        # Predict and evaluate
        y_pred = model.predict(X_val)
        if error_metric == 'mse':
            error = mean_squared_error(y_val_values, y_pred)
        elif error_metric == 'mae':
            error = mean_absolute_error(y_val_values, y_pred)
        elif error_metric == 'rmse':
            error = root_mean_squared_error(y_val_values, y_pred)
        else:
            raise ValueError("Error metric must be 'mse' or 'mae'.")
        errors.append(error)
        models.append(model)
    return errors, models

def plot_expanding_window_splits_per_iteration(df_train_val, mse_scores, y_targets, tscv):
    """
    Plots one figure per expanding window cross-validation split, showing both training and validation sets.
    
    Parameters:
    df_train_val (pd.DataFrame): The dataset containing the training and validation data with a time index.
    mse_scores (list): List of MSE scores corresponding to each expanding window iteration.
    initial_train_size (int): The size of the initial training window.
    val_size (int): The size of the validation window.
    step_size (int): The size of steps by which the training window expands.
    """
    iteration = 0
    for train_index, test_index in tscv.split(df_train_val):
        iteration += 1
        train_range = train_index#df_train_val.index[:i]
        val_range = test_index#df_train_val.index[i:i + val_size]
        
        # Create a new plot for each split
        fig, ax = plt.subplots(figsize=(8, 6))
        
        # Plot the entire time series (target variable)
        ax.plot(df_train_val.index, df_train_val[y_targets[0]], label="Time Series", color='black', alpha=0.5)
        
        # Highlight training window
        ax.axvspan(train_range[0], train_range[-1], color='lightblue', alpha=0.6, label='Training Set')
        
        # Highlight validation window
        ax.axvspan(val_range[0], val_range[-1], color='lightgreen', alpha=0.6, label='Validation Set')
        
        # Display MSE score
        mse_score = mse_scores[iteration - 1]  # Get corresponding MSE score for this split
        ax.text(0.5, 0.9, f"Error: {mse_score:.4f}", horizontalalignment='center', verticalalignment='center', transform=ax.transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.6))
        
        # Set plot labels and title
        ax.set_title(f'Expanding Window Cross-Validation Split {iteration}', fontsize=14)
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Target Variable', fontsize=12)
        ax.legend(loc='upper left')
        
        # Show the plot
        plt.show()

def forward_feature_selection(feature_groups, validation_method, df_train_val, df_train, df_val, df_test, y_targets, tscv):
    '''
    - feature_groups: a dictionary with group names as keys and lists of column names as values.
    - validation_method: a string specifying the validation method to use.
        - "expanding_window"
        - "naive" - just use validation set
    '''
    selected_groups = []
    remaining_groups = list(feature_groups.keys())  # list of group names (keys)
    best_error = np.inf

    # document the process
    selection_history = []

    round_counter = 0
    print("\nStarting forward feature selection process...")
    while remaining_groups:
        round_counter += 1
        print(f"Round {round_counter} - selected groups: {selected_groups}")
        best_group = None
        for group in remaining_groups:
            # Combine selected groups with the current group (concatenate column names)
            current_columns = list(set(sum([feature_groups[g] for g in (selected_groups + [group])], [])))
            #print(f"- trying group: {group} --- Current Cols: {current_columns}\n\n")

            # create datasets with new columns
            df_train_val_copy = df_train_val.copy()
            X_train, y_train = df_train[current_columns], df_train[y_targets]
            X_val, y_val = df_val[current_columns], df_val[y_targets]
            if len(y_targets) == 1: 
                y_train_values = y_train.values.ravel()
            else:
                y_train_values = y_train.values        
            # Train and evaluate using cross-validation (or any other metric)
            if validation_method == "naive":
                model = RandomForestRegressor(random_state=42, n_estimators=200)
                model.fit(X_train, y_train_values)
                y_pred = model.predict(X_val)
                error = root_mean_squared_error(y_val, y_pred)
            elif validation_method == "expanding_window":
                model = RandomForestRegressor(random_state=42, n_estimators=200)
                errors, _ = expanding_window_cv(df_train_val = df_train_val_copy, model = model,
                                                x_features = current_columns, y_targets = y_targets, 
                                                error_metric = 'rmse', tscv = tscv)
                error = np.mean(errors)
            
            if error < best_error:
                best_error = error
                best_group = group
        
        if best_group is not None:
            # Add the best group to selected and remove from remaining
            selected_groups.append(best_group)
            remaining_groups.remove(best_group)
            
            # Document the step
            selection_history.append({
                'step': len(selected_groups),
                'added_group': best_group,
                'rmse_error': best_error,
                'selected_groups': selected_groups.copy()
            })
        else:
            # No improvement, stop the process
            break

    # Convert the history to a DataFrame for easier documentation
    results_df = pd.DataFrame(selection_history)
    best_features = list(set(sum([feature_groups[g] for g in selected_groups], [])))
    return results_df, best_features

def grid_seach_hyperparameters(X_train_val, y_train_val, tscv):
    '''
    Perform grid search to find the best hyperparameters for the RandomForestRegressor model.
    '''
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7, 10, 12],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4, 8],
        'max_features': ['sqrt', 'log2', 0.5],
        'bootstrap': [True, False]
    }
    num_cores = os.cpu_count()
    print("\nGrid search with cross-validation...")
    print(f"Number of CPU cores available: {num_cores}")

    # Initialize the RandomForestRegressor
    rf = RandomForestRegressor(random_state=42)

    # Perform grid search with cross-validation
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=tscv, scoring='neg_root_mean_squared_log_error', verbose=1, n_jobs=num_cores-4)
    grid_search.fit(X_train_val, y_train_val)
    return grid_search

def save_and_plot_grid_results(grid_results, path, tscv):
    # write txt file with best params and scores
    txt_str = ""
    txt_str += f"Best Parameters: {grid_results.best_params_}\n"
    cv_results = grid_results.cv_results_
    # Find the best index
    best_index = grid_results.best_index_
    # Extract split scores for the best estimator
    best_split_scores = [cv_results[f'split{i}_test_score'][best_index] for i in range(tscv.n_splits)]
    txt_str += f"Best split scores for the best estimator:{best_split_scores}\n"

    with open(os.path.join(path, 'grid_search_info.txt'), 'w') as f:
        f.write(txt_str)


    # Convert cv_results_ to a DataFrame
    df_grid_results = pd.DataFrame(grid_results.cv_results_)
    # show total cell width
    pd.set_option('display.max_colwidth', None)
    df_grid_results[['rank_test_score', 'params', 'mean_test_score', 'std_test_score']].sort_values(by='rank_test_score').head(10)
    df_grid_results.to_csv(os.path.join(path, "grid_search_results.csv"), index=False)
    
    
    # vizualize grid search
    pivot_table = df_grid_results.pivot_table(values='mean_test_score', index='param_max_depth', columns='param_n_estimators')
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot_table, annot=True, fmt=".3f", cmap="YlGnBu")
    plt.title("Mean Test Score for 'max_depth' vs 'n_estimators'")
    plt.xlabel("n_estimators")
    plt.ylabel("max_depth")
    # save plot
    plt.savefig(os.path.join(path, "grid_search_heatmap.png"))

    # Example Line Plot: Visualize the effect of 'max_depth' on mean test score
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df_grid_results, x="param_max_depth", y="mean_test_score", marker="o")
    plt.title("Mean Test Score by 'max_depth'")
    plt.xlabel("Max Depth")
    plt.ylabel("Mean Test Score")
    plt.savefig(os.path.join(path, "grid_search_lineplot.png"))

    
    feature_importance_df = pd.DataFrame({
        'Feature': grid_results.best_estimator_.feature_names_in_,
        'Importance': grid_results.best_estimator_.feature_importances_
    })

    # export feature importance to csv
    feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
    # export to csv
    feature_importance_df.to_csv(os.path.join(path, "feature_importance.csv"), index=False)

    # plot feature importance
    plt.figure(figsize=(12, 6))
    sns.barplot(data=feature_importance_df, x='Importance', y='Feature', palette='viridis')
    plt.title('Feature Importances')
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.tight_layout()  # Automatically adjusts the plot layout
    plt.savefig(os.path.join(path, "feature_importance_plot.png"))

def get_predictions(model, X_test, y_test, horizons=[1, 3, 5, 7], y_target_types=['close']):
    '''
    Get predictions for multiple forecast horizons using a trained model and the test set.
    '''
    print("\nGenerating predictions for multiple forecast horizons...")
    all_actuals = {h: [] for h in horizons}
    all_predictions = {h: [] for h in horizons}
    all_errors = {h: [] for h in horizons}
    test_results_dict = {h: [] for h in horizons}


    # Iterate over each horizon
    for horizon in horizons:
        predictions = []
        actuals = []
        errors = []
        for i in range(len(X_test) - horizon):
            # Start with the current test row
            X_test_tmp = X_test.copy()
            
            # Initialize prediction list for this row and horizon
            row_predictions, row_actuals = [], []
            
            for day in range(0, horizon):
                test_row = X_test_tmp.iloc[i + day].copy()
                # Predict for the next day
                if len(y_target_types) == 1:
                    y_pred = model.predict(test_row.values.reshape(1, -1))[0]
                    y_actual = y_test.iloc[i + day].values[0]

                else:
                    y_pred = model.predict(test_row.values)
                    y_actual = y_test.iloc[i + day].values

                row_predictions.append(y_pred)
                row_actuals.append(y_actual)
                
                if len(y_target_types) == 1:
                    # Replace actual value with the predicted value for the next day's prediction
                    X_test_tmp[f"-10|ENV__SOIL__VWC|{y_target_types[0]}"].iloc[i+day+1] = y_pred
                else:
                    for y_target_type, y_pred_target_type in zip(y_target_types, y_pred):
                        # Replace actual value with the predicted value for the next day's prediction
                        X_test_tmp[f"-10|ENV__SOIL__VWC|{y_target_type}"].iloc[i+day+1] = y_pred_target_type
                
            # Store predictions for this horizon
            predictions.append(row_predictions)
            actuals.append(row_actuals)


        # compute root mean squared error
        # flatten the list of predictions and actuals
        predictions_flat = [item for sublist in predictions for item in sublist]
        actuals_flat = [item for sublist in actuals for item in sublist]
        errors = root_mean_squared_error(actuals_flat, predictions_flat)
        all_predictions[horizon] = predictions
        all_actuals[horizon] = actuals
        all_errors[horizon] = errors
        test_results_dict[horizon] = {
            'predictions': predictions,
            'actuals': actuals,
            'errors': errors
        }
    return test_results_dict

def plot_predictions(df, index, horizon, y_pred, y_actual, rmse, path, y_target_types=['close']):
    """
    Plot the last 5 days of actual data and the predicted vs. actual values for the forecast horizon.
    """
    # Precipitation and irrigation data for the last days + forecast period
    full_series_precipitation = df.loc[index, ['precipitation|sum|D-4', 'precipitation|sum|D-3', 'precipitation|sum|D-2', 'precipitation|sum|D-1', 'precipitation|sum'] + [f'precipitation|sum|D+{d}' for d in range(1, horizon+1)]].values
    full_series_irrigation = df.loc[index, ['DOC_IRRIGATION_PREDICTION|max|D-4', 'DOC_IRRIGATION_PREDICTION|max|D-3', 'DOC_IRRIGATION_PREDICTION|max|D-2', 'DOC_IRRIGATION_PREDICTION|max|D-1', 'DOC_IRRIGATION_PREDICTION|max'] + [f'DOC_IRRIGATION_PREDICTION|max|D+{d}' for d in range(1, horizon+1)]].values * 100
    
    # Time index including past and forecast days
    time_index = np.arange(-4, horizon + 1)

    fig, ax1 = plt.subplots(figsize=(10, 6))
    # Extract the last 5 days of soil moisture and future days based on horizon
    if len(y_target_types) == 1:
        y_target_type= y_target_types[0]
        last_days_vwc = df.loc[index, [f'-10|ENV__SOIL__VWC|{y_target_type}|D-4', f'-10|ENV__SOIL__VWC|{y_target_type}|D-3', f'-10|ENV__SOIL__VWC|{y_target_type}|D-2', f'-10|ENV__SOIL__VWC|{y_target_type}|D-1', f'-10|ENV__SOIL__VWC|{y_target_type}']].values
        actual_full_series_vwc = np.concatenate([last_days_vwc, y_actual])
        predicted_series_vwc = np.concatenate([last_days_vwc, y_pred])

        # Plot setup
        ax1.plot(time_index, predicted_series_vwc, label='Predicted', marker='x', color='red')
        ax1.plot(time_index, actual_full_series_vwc, label='Actual', marker='o', color='green')

    elif y_target_types == ['min', 'max']:  
        last_days_vwc_min = df.loc[index, ['-10|ENV__SOIL__VWC|min|D-4', '-10|ENV__SOIL__VWC|min|D-3', '-10|ENV__SOIL__VWC|min|D-2', '-10|ENV__SOIL__VWC|min|D-1', '-10|ENV__SOIL__VWC|min']].values
        last_days_vwc_max = df.loc[index, ['-10|ENV__SOIL__VWC|max|D-4', '-10|ENV__SOIL__VWC|max|D-3', '-10|ENV__SOIL__VWC|max|D-2', '-10|ENV__SOIL__VWC|max|D-1', '-10|ENV__SOIL__VWC|max']].values
        t, t1 = y_actual[:horizon], y_actual[horizon:]
        actual_full_series_vwc_min = np.concatenate([last_days_vwc_min, np.asarray(y_actual[:horizon])])
        actual_full_series_vwc_max = np.concatenate([last_days_vwc_max, np.asarray(y_actual[horizon:])])
        predicted_series_vwc_min = np.concatenate([last_days_vwc_min, np.asarray(y_pred[:horizon])])
        predicted_series_vwc_max = np.concatenate([last_days_vwc_max, np.asarray(y_pred[horizon:])])

        # Plot setup
        ax1.plot(time_index, predicted_series_vwc_min, label='Predicted Min', color='red', linestyle='--')
        ax1.plot(time_index, predicted_series_vwc_max, label='Predicted Max', color='red', marker='o')
        ax1.plot(time_index, actual_full_series_vwc_min, label='Actual Min', linestyle='--', color='green')
        ax1.plot(time_index, actual_full_series_vwc_max, label='Actual Max', marker='o', color='green')
        #ax1.fill_between(pred_time_index, actual_full_series_vwc_min, actual_full_series_vwc_max, color='green', alpha=0.2, label='Actual Range')
        #ax1.fill_between(pred_time_index, predicted_series_vwc_min, predicted_series_vwc_max, color='red', alpha=0.05, label='Pred Range')
                                                       
    ax1.set_xlabel('Time (days)')
    ax1.set_ylabel('Soil VWC', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.axvline(x=0, color='gray', linestyle='--', label='Current Day')
    # Plot water intake
    ax2 = ax1.twinx()
    ax2.bar(time_index, full_series_precipitation, width=0.1, align='edge', color='blue', alpha=0.6, label='Precipitation')
    ax2.bar(time_index, full_series_irrigation, width=0.1, align='edge', color='gray', alpha=0.6, label='Irrigation')
    ax2.tick_params(axis='y', labelcolor='blue')
    ax2.set_ylim(0, 110)

    # Legends and title
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.title(f'{horizon}-day Forecast for Test Index {index} - RMSE: {rmse:.2f}  - {y_target_types}')
    
    # Save plot
    # join path with folder same name as horizon
    path = os.path.join(path, f"{horizon}")
    # make directory if not exists
    if not os.path.exists(path): os.makedirs(path)
    safe_index = str(index).replace(":", "-").replace(" ", "_")
    plt.savefig(os.path.abspath(os.path.join(path, f"prediction_index_{safe_index}.png")))
    plt.close()

def plot_min_max_predictions(df, index, horizon, y_pred, y_actual, rmse, path):
    """
    Plot the last 5 days of actual data and the predicted vs. actual values for the forecast horizon.
    """
    # Extract the last 5 days of soil moisture and future days based on horizon
    last_days_vwc = df.loc[index, ['-10|ENV__SOIL__VWC|min|D-4', '-10|ENV__SOIL__VWC|min|D-3', '-10|ENV__SOIL__VWC|min|D-2', '-10|ENV__SOIL__VWC|min|D-1', '-10|ENV__SOIL__VWC|min']].values
    
    actual_full_series_vwc = np.concatenate([last_days_vwc, y_actual])
    predicted_series_vwc = np.concatenate([last_days_vwc, y_pred])
    
    # Precipitation and irrigation data for the last days + forecast period
    full_series_precipitation = df.loc[index, ['precipitation|sum|D-4', 'precipitation|sum|D-3', 'precipitation|sum|D-2', 'precipitation|sum|D-1', 'precipitation|sum'] + [f'precipitation|sum|D+{d}' for d in range(1, horizon+1)]].values
    full_series_irrigation = df.loc[index, ['DOC_IRRIGATION_PREDICTION|max|D-4', 'DOC_IRRIGATION_PREDICTION|max|D-3', 'DOC_IRRIGATION_PREDICTION|max|D-2', 'DOC_IRRIGATION_PREDICTION|max|D-1', 'DOC_IRRIGATION_PREDICTION|max'] + [f'DOC_IRRIGATION_PREDICTION|max|D+{d}' for d in range(1, horizon+1)]].values * 100
    
    # Time index including past and forecast days
    time_index = np.arange(-4, horizon + 1)

    # Plot setup
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(time_index, predicted_series_vwc, label='Predicted', marker='x', color='red')
    ax1.plot(time_index, actual_full_series_vwc, label='Actual', marker='o', color='green')
    ax1.set_xlabel('Time (days)')
    ax1.set_ylabel('Soil VWC', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    ax1.axvline(x=0, color='gray', linestyle='--', label='Current Day')

    # Plot water intake
    ax2 = ax1.twinx()
    ax2.bar(time_index, full_series_precipitation, width=0.1, align='edge', color='blue', alpha=0.6, label='Precipitation')
    ax2.bar(time_index, full_series_irrigation, width=0.1, align='edge', color='gray', alpha=0.6, label='Irrigation')
    ax2.tick_params(axis='y', labelcolor='blue')
    ax2.set_ylim(0, 110)

    # Legends and title
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.title(f'{horizon}-day Forecast for Test Index {index} - RMSE: {rmse:.2f}')
    
    # Save plot
    # join path with folder same name as horizon
    path = os.path.join(path, f"{horizon}")
    safe_index = str(index).replace(":", "-").replace(" ", "_")
    plt.savefig(os.path.abspath(os.path.join(path, f"prediction_index_{safe_index}.png")))
    plt.close()

def create_plot_predictions(df, X_test, test_results_dict, path):
    for horizon, horizon_results in test_results_dict.items():
        predictions = horizon_results['predictions']
        actuals = horizon_results['actuals']
        errors = horizon_results['errors']

        for index in range(len(predictions)):
            y_preds = predictions[index]
            y_actuals = actuals[index]
            rmse_errors = root_mean_squared_error(y_actuals, y_preds)
            plot_predictions(df, X_test.index[index], horizon, y_preds, y_actuals, rmse_errors, path)

def analyze_predictions(df, X_test, y_preds, y_actuals, run_path, y_target_types, days_ahead, y_targets):
    print("\nAnalyzing predictions...")
    print("Plotting predictions...")
    # create plots
    for index in range(len(y_preds)):
        y_pred = np.array(y_preds[index]).ravel()
        y_actual = np.array(y_actuals[index]).ravel()

        rmse_errors = root_mean_squared_error(y_actual, y_pred)

        plot_predictions(df, X_test.index[index], days_ahead, y_pred, y_actual, rmse_errors, run_path, y_target_types=y_target_types)
        
    def calculate_errors(y_true, y_pred):
        # Calculate errors
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        return mse, rmse, mae, mape
    
    print("\nCalculating errors...")
    # Calculate errors for the entire test set
    mse, rmse, mae, mape = calculate_errors(y_actuals.ravel(), y_preds.ravel())
    # convert to df
    errors_df = pd.DataFrame({
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape
    }, index=['Total'])
    # save to csv
    errors_df.to_csv(os.path.join(run_path, 'errors_total.csv'))

    # Plotting the errors
    plt.figure()
    errors_df.plot(kind='bar', figsize=(12, 6))
    plt.title('Error Metrics')
    plt.ylabel('Error Value')
    #shift xticks to left
    plt.gca().margins(x=0)
    plt.tight_layout()
    plt.savefig(os.path.join(run_path, 'errors_total_plot.png'))


    # Calculate errors per variable
    if len(y_targets) > 1:
        errors_per_variable = {}
        for i, (y_target, y_true, y_pred) in enumerate(zip(y_targets, y_actuals.T, y_preds.T)):
            mse, rmse, mae, mape = calculate_errors(y_true, y_pred)
            errors_per_variable[f'Variable {y_target}'] = {
                'MSE': mse,
                'RMSE': rmse,
                'MAE': mae,
                'MAPE': mape
            }
        
        # Convert errors to a DataFrame
        errors_df = pd.DataFrame(errors_per_variable).T
        # Save errors to a CSV file
        errors_df.to_csv(os.path.join(run_path, 'errors_per_variable.csv'))

        # Plot only RMSE
        plt.figure()
        errors_df['RMSE'].plot(kind='bar', figsize=(12, 6))
        plt.title('RMSE per Predicted Variable')
        plt.ylabel('RMSE Value')
        plt.xlabel('Predicted Variable')
        plt.xticks(rotation=45, ha='right')  
        plt.tight_layout()
        plt.savefig(os.path.join(run_path, 'rmse_per_variable_plot.png'))
        
        # Plotting the errors
        plt.figure()
        errors_df.plot(kind='bar', figsize=(12, 6))
        plt.title('Error Metrics per Predicted Variable')
        plt.ylabel('Error Value')
        plt.xlabel('Predicted Variable')
        plt.xticks(rotation=45, ha='right')  
        #shift xticks to left
        plt.gca().margins(x=0)
        plt.tight_layout()
        plt.savefig(os.path.join(run_path, 'errors_per_variable_plot.png'))

def save_predictions(y_preds, y_actuals, y_targets, df_test, run_path):
    # save y_preds and y_actuals + corresponding columnnames + dates to json with structure:
    # {data: {y_preds: [], y_actuals: [], y_column_names: [], dates: []}}
    data = {
        'y_preds': y_preds.tolist(),
        'y_actuals': y_actuals.tolist(),
        'y_column_names': y_targets,
        'dates': df_test.index.strftime('%Y-%m-%d %H:%M:%S').tolist()  # Convert datetime to string
    }
    with open(os.path.join(run_path, 'predictions.json'), 'w') as f:
        json.dump(data, f)

def create_run_folder(experiment_name):
    # create folder named "run_" + name of the experiment + counter if exists
    current_dir = os.getcwd()
    print(current_dir)
    path = os.path.join(current_dir, '../../results/runs') 
    current_date_time = datetime.now().strftime("%Y-%m-%d_%H-%M")
    folder_name = f"run_{experiment_name}_{current_date_time}"
    path = os.path.join(path, folder_name)
    # check if folder exists
    if not os.path.exists(path):
        os.makedirs(path)
        print(f"Saving results to: {path}\n")
    else:
        print(f"Error")
        raise ValueError(f"Folder {folder_name} already exists.")
    return path

def run_experiment(EXPERIMENT_NAME, DAYS_AHEAD, Y_TARGET_TYPES, INCLUDE_IRRIGATION_EVENTS, VALIDATION_METHOD, DATA_GRANULARITY):
    # Create a folder to save the
    run_path = create_run_folder(EXPERIMENT_NAME)

    # Load the data
    if DATA_GRANULARITY == 'd':
        df_full = load_daily_data()
    elif DATA_GRANULARITY == 'h':
        df_full = load_hourly_data()
    else:
        raise ValueError("Data granularity must be 'd' or 'h'.")
    df = df_full.copy()

    # feature groups
    feature_groups, y_targets = load_feature_groups(num_days_ahead_pred=DAYS_AHEAD, y_target_types=Y_TARGET_TYPES, data_granularity=DATA_GRANULARITY)
    features = get_features_from_groups(df, feature_groups)

    #print(f"Feature Groups: {feature_groups}\n")
    print(f"Targets ({len(y_targets)}): {y_targets}\n")
    df = df[features + y_targets]

    if not INCLUDE_IRRIGATION_EVENTS:
        rows_before = df.shape[0]
        # Identify columns D and D+ days
        columns_to_exclude = [col for col in df.columns if col.startswith('DOC_IRRIGATION_PREDICTION|')]
        # Filter out rows where any of these columns have a value of 1
        df = df[~df[columns_to_exclude].any(axis=1)]
        rows_after = df.shape[0]
        print(f"Removed {rows_before - rows_after} rows with irrigation prediction values.")
        # write to file
        with open(os.path.join(run_path, 'irrigation_events_removed.txt'), 'w') as f:
            f.write(f"Removed {rows_before - rows_after} rows with irrigation prediction values.")

    # Drop rows with missing values
    with open("missing_values.txt", "w") as file:
        # Print and write the first statement
        text = "\nDrop rows with missing values"
        print(text)
        file.write(text + "\n")
        missing_values = f'Missing values in train data: {df.isnull().sum().sum()}'
        print(missing_values)
        file.write(missing_values + "\n")
        missing_rows_index = f'Missing values in train data: {df[df.isnull().any(axis=1)].index.tolist()}'
        print(missing_rows_index)
        file.write(missing_rows_index + "\n")
        df = df.dropna()
        missing_values_after_drop = f'Missing values in train data after drop: {df.isnull().sum().sum()}'
        print(missing_values_after_drop)
        file.write(missing_values_after_drop + "\n")

    # split dataset
    df_train_val, df_train, df_val, df_test = split_dataset(df, train_start='2023-05-22', val_start='2024-05-02', test_start='2024-08-01', days_ahead=DAYS_AHEAD, path=run_path)

    # get dates from datetime column
    df_train_val["dates"] = df_train_val.index.date


    # Define Cross Validation Splits: Use expanding window approach using TimeSeriesSplit
    # - chose parameters such that we train at least til Mar 24 and validate only on data in 2024 during irrigation season bcs we want to optimize for this season
    # - validation sets are always 30 days
    # optimize the split such that the validation happens only on data in the irrigation season
    
    TSCV = TimeSeriesSplit(gap=0, max_train_size=None, n_splits=3, test_size=30) 
    tscv_info = "TimeSeriesSplit(gap=0, max_train_size=None, n_splits=4, test_size=30)\nSplits:\n"
    for train_index, test_index in TSCV.split(df_train_val):
        print("\nCV TS Split: ")
        train_data = df_train_val.iloc[train_index]
        test_data = df_train_val.iloc[test_index]
        
        # Optional: Print dates for each split to verify the periods
        tscv_info += f"Train period: {train_data['dates'].min()} - {train_data['dates'].max()}\n"
        tscv_info += f"Test period: {test_data['dates'].min()} - {test_data['dates'].max()}\n"
        tscv_info += "-"* 50 + "\n"
    print(tscv_info)
    # save to file
    with open(os.path.join(run_path, 'tscv_info.txt'), 'w') as f:
        f.write(tscv_info)
    

    # Forward Feature Selection
    df_forward_selection_report, best_features = forward_feature_selection(feature_groups=feature_groups, validation_method=VALIDATION_METHOD, df_train_val=df_train_val, df_train=df_train, df_val=df_val, df_test=df_test, y_targets=y_targets, tscv=TSCV)
    # save report under src model
    df_forward_selection_report.to_csv(os.path.join(run_path, 'forward_selection_report.csv'), index=False)
    
    X_train_val, y_train_val = df_train_val[best_features], df_train_val[y_targets]
    if len(y_targets) == 1: 
        y_train_val_values = y_train_val.values.ravel()
    else:
        y_train_val_values = y_train_val.values
    # Grid Search for Hyperparameters
    grid_search_results = grid_seach_hyperparameters(X_train_val, y_train_val_values, tscv=TSCV)

        
    best_rf_model = grid_search_results.best_estimator_
    # get scores of gridsearch per cross-validation split
    save_and_plot_grid_results(grid_search_results, path=run_path, tscv=TSCV)


    # Test the model on the test set
    X_test = df_test[best_features]
    y_test = df_test[y_targets]

    y_preds = best_rf_model.predict(X_test)
    y_actuals = y_test.values
    save_predictions(y_preds, y_actuals, y_targets, df_test, run_path)
    analyze_predictions(df_full, X_test, y_preds, y_actuals, run_path, Y_TARGET_TYPES, DAYS_AHEAD, y_targets)

    # save model
    with open(os.path.join(run_path, 'model.pkl'), 'wb') as f:
        pickle.dump(best_rf_model, f)

if __name__ == "__main__":

    # Define experiment parameters
    DATA_GRANULARITY_OPTIONS = [
        'h', 
        'd'
        ]
    DAYS_AHEAD_OPTIONS = [
        1, 
        3, 
        5, 
        7
        ]
    Y_TARGET_TYPES_OPTIONS = {
        'min': ['min'],                # For daily, only min
        'minmax': ['min', 'max'],  # For daily, both minmax and close
        'close': ['close']             # For hourly, only close
    }
    INCLUDE_IRRIGATION_EVENTS = [
        True, 
        False
    ]
    VALIDATION_METHOD = "expanding_window"

    # Loop through each parameter configuration
    for data_granularity in DATA_GRANULARITY_OPTIONS:
        for days_ahead in DAYS_AHEAD_OPTIONS:
            for y_target_type in list(Y_TARGET_TYPES_OPTIONS.keys()):
                for include_irrigations in INCLUDE_IRRIGATION_EVENTS:
                    # Experiment name for clarity in results
                    experiment_name = f'{days_ahead}days_{data_granularity}_{include_irrigations}Irrigations_{y_target_type}'
                    print(f"\nRunning experiment: {experiment_name}\nConfiguration: {days_ahead} days ahead, {data_granularity} data granularity, {include_irrigations} irrigation prediction, {y_target_type} target type")
                    print("-" * 50)
                    t = Y_TARGET_TYPES_OPTIONS[y_target_type]
                    # Run the experiment with the current configuration
                    run_experiment(
                        EXPERIMENT_NAME=experiment_name,
                        DAYS_AHEAD=days_ahead,
                        Y_TARGET_TYPES=Y_TARGET_TYPES_OPTIONS[y_target_type],
                        INCLUDE_IRRIGATION_EVENTS=include_irrigations,
                        VALIDATION_METHOD=VALIDATION_METHOD,
                        DATA_GRANULARITY=data_granularity
                    )