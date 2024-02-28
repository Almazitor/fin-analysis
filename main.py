# UNCOMMENT THE LINE 3 THROUGH 9 TO INSTALL LIBRARIES

# import subprocess
# import sys
#
# # Install required libraries
# libraries = ['pandas', 'matplotlib', 'numpy', 'scipy', 'statsmodels', 'datetime', 'sklearn', 'seaborn', 'alpha_vantage']
# for lib in libraries:
#     subprocess.check_call([sys.executable, '-m', 'pip', 'install', lib])

# Import the newly installed libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import warnings
import itertools
import datetime
import os
import seaborn as sns
from datetime import datetime
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from alpha_vantage.timeseries import TimeSeries

# Disable warnings for better clarity
warnings.filterwarnings("ignore")

# Setting pandas to display all columns
pd.set_option('display.max_columns', None)

def get_AV_api_key():
    while True:
        api_key = input('Enter your Alpha Vantage API key: ')
        # Add a condition to check the validity of the API key
        if api_key:
            return api_key
        else:
            print("API key cannot be empty. Please enter a valid API key.")

def load_and_filter_stock_data(stock_abbreviation, api_key):
    while True:
        try:
            ts = TimeSeries(key=api_key, output_format='pandas')
            data, meta_data = ts.get_daily(symbol=stock_abbreviation, outputsize='full')

            # Convert 'Date' column to datetime format
            data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            data.index = pd.to_datetime(data.index)

            # Filter data by date range (3 years before the last date)
            end_date = data.index.max()
            start_date = end_date - pd.DateOffset(years=3)

            df_filtered = data[(data.index >= start_date) & (data.index <= end_date)]
            df_filtered = df_filtered.asfreq('B')

            # Check if the data is suitable for analysis
            if len(df_filtered) < 3 * 252:  # Assuming 252 trading days in a year
                raise ValueError('Data is not suitable for analysis. Less than three years of data available.')

            return df_filtered

        except FileNotFoundError:
            print(f'Error: Could not find data for {stock_abbreviation}. Please check the file path.')
            return None
        except ValueError as e:
            print(f'Error: {e}')
            stock_abbreviation = input('Enter the correct stock abbreviation (e.g., AAPL): ')
            continue  # Restart the loop with the new stock abbreviation

# Ask user for stock abbreviation
while True:
    stock_abbreviation = input('Enter the stock abbreviation (e.g., AAPL): ')
    alpha_vantage_api_key = get_AV_api_key()
    # Load and filter stock data
    df = load_and_filter_stock_data(stock_abbreviation, alpha_vantage_api_key)
    if df is not None:
        df = df.bfill()
        print(df.head())
        break  # Break the loop if data is successfully loaded
    else:
        print("Error loading and filtering stock data. Please try again.")

def evaluate_stationarity(data):
    data = data.bfill()
    # Perform first-order differencing
    differenced_data = data.diff().dropna()
    # Drop any remaining NaN values after differencing
    differenced_data = differenced_data.dropna()

    # Perform ADF test and KPSS test on differenced data
    adf_result_diff = adfuller(differenced_data)
    kpss_result_diff = kpss(differenced_data)

    # Check if differenced data is stationary
    is_stationary_diff = adf_result_diff[1] < 0.05 and kpss_result_diff[1] > 0.05

    # Interpret the results
    if is_stationary_diff:
        print("Differenced data is stationary.")
        return differenced_data
    else:
        print("Differenced data is not stationary. Results shall be less accurate.")
        return differenced_data

def find_best_arima_parameters_and_forecast(df_stationary):
    global df

    # Replace NaN values with backward fill
    df_stationary = df_stationary.bfill()

    train_size = len(df_stationary) - 10

    train, test = df_stationary[:train_size], df_stationary[train_size:]

    forecast_steps = len(test)
    actual_values = df['Close'].iloc[train_size + 1:]

    # Calculate the variance of the original data for the testing period
    variance_test_data = np.var(actual_values)
    print(f'Variance of the testing period: {variance_test_data}')

    # Initialize variables to store the best combination and corresponding MSE
    best_p = None
    best_q = None
    best_mse = float('inf')  # Initialize with a large value

    significant_lags_p = [i for i in range(21)]
    significant_lags_q = [i for i in range(21)]

    # Loop through all combinations of p and q
    for p, q in itertools.product(significant_lags_p, significant_lags_q):
        try:
            # Train ARIMA model
            order_ARIMA = (p, 0, q)
            model_ARIMA = ARIMA(train, order=order_ARIMA)
            fit_ARIMA = model_ARIMA.fit()

            # Forecast future values for the testing period
            forecast_ARIMA = fit_ARIMA.get_forecast(steps=forecast_steps)
            forecast_values_ARIMA = forecast_ARIMA.predicted_mean

            # Inverse difference the forecasted values
            forecast_original_ARIMA = df['Close'].iloc[train_size - 1] + forecast_values_ARIMA.cumsum()

            # Calculate MSE for the testing period
            mse = mean_squared_error(actual_values, forecast_original_ARIMA)

            # Check if the current combination has a lower MSE
            if mse < best_mse:
                best_p = p
                best_q = q
                best_mse = mse
                if best_mse < variance_test_data:
                    break
        except Exception as e:
            print(f'Error for combination {order_ARIMA}: {str(e)}')
            continue

    # Print the best combination
    print(f'Best (p, 0, q) combination: ({best_p}, 0, {best_q}) with MSE: {best_mse}')

    # Compare MSE with the variance
    if best_mse < variance_test_data:
        print("Model determined the best fit.")
    else:
        print("Model was not able to determine the best fit. Results shall be less accurate.")

    return best_p, best_q

def forecast_and_plot(best_p, best_q,df_stationary):
    # Forecast for the next 10 days
    forecast_steps = 10
    order_ARIMA = (best_p, 0, best_q)

    # Train ARIMA model using all available data
    model_ARIMA = ARIMA(df_stationary, order=order_ARIMA)
    fit_ARIMA = model_ARIMA.fit()

    # Forecast future values for the next 10 days
    forecast_ARIMA = fit_ARIMA.get_forecast(steps=forecast_steps)
    forecast_values_ARIMA = forecast_ARIMA.predicted_mean

    # Inverse difference the forecasted values
    forecast_original_ARIMA = df['Close'].iloc[-1] + forecast_values_ARIMA.cumsum()

    # Plot the results for the next 10 days
    plt.figure(figsize=(18, 9))
    plt.plot(forecast_original_ARIMA.index, forecast_original_ARIMA, color='blue',
             label=f'Forecast (ARIMA) - (p, 0, q) = ({best_p}, 0, {best_q})')

    # Plot confidence intervals
    confidence_intervals_ARIMA = forecast_ARIMA.conf_int()
    plt.fill_between(forecast_original_ARIMA.index,
                     forecast_original_ARIMA + confidence_intervals_ARIMA.iloc[:, 0].cumsum(),
                     forecast_original_ARIMA + confidence_intervals_ARIMA.iloc[:, 1].cumsum(),
                     color='pink', alpha=0.3, label='Confidence Intervals')

    plt.legend()
    plt.title('ARIMA Model Forecast for Next 10 Days')

    save_plot_with_choice(plt, 'Analysis_Results', 'Forecast_Plot_10days.png')
    plt.show()

    # Create a table with forecasted values and dates
    forecast_table = pd.DataFrame({'Date': forecast_original_ARIMA.index,
                                   'Forecasted Price': forecast_original_ARIMA.values})
    forecast_table.set_index('Date', inplace=True)

    # Print the table
    print("Forecasted Prices for Next 10 Days:")
    print(forecast_table)

    # Ask the user if they want to save the table
    save_table_choice = input("Do you want to save the table? (y/n): ").lower()

    # Create a folder on the desktop
    desktop_path = os.path.join(os.path.expanduser('~'), 'Desktop')
    folder_name = 'Analysis_Results'
    folder_path = os.path.join(desktop_path, folder_name)

    # Check if the folder exists, if not, create it
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    if save_table_choice == 'y':
        # Save the table inside the folder
        table_path = os.path.join(folder_path, 'ARIMA_Forecast_Table.txt')
        forecast_table.to_csv(table_path, sep='\t')
        print(f'Table has been saved in the folder "{folder_name}" as "ARIMA_Forecast_Table.txt".')

def format_date_input(input_date):
    try:
        formatted_date = datetime.strptime(input_date, "%Y%m%d").strftime("%Y-%m-%d")
        return formatted_date
    except ValueError:
        print("Error: Please enter a valid date in YYYYMMDD format.")
        return None

def create_analysis_dataframe(alpha_vantage_api_key, stock_abbreviation):
    while True:
        # Ask user for start and end dates
        start_date_input = input('Enter the start date in YYYYMMDD format: ')
        end_date_input = input('Enter the end date in YYYYMMDD format: ')

        # Format user input for start date
        start_date = format_date_input(start_date_input)
        if start_date is None:
            continue  # Restart the loop if the date is not valid

        # Format user input for end date
        end_date = format_date_input(end_date_input)
        if end_date is None:
            continue  # Restart the loop if the date is not valid

        ts = TimeSeries(key=alpha_vantage_api_key, output_format='pandas')
        data, meta_data = ts.get_daily(symbol=stock_abbreviation, outputsize='full')

        # Convert 'Date' column to datetime format
        data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
        data.index = pd.to_datetime(data.index)

        # Filter data by date range
        df_analysis = data[(data.index >= start_date) & (data.index <= end_date)]
        df_analysis = df_analysis.asfreq('B')

        return df_analysis

# Ask user for stock abbreviation
while True:
    # Create analysis dataframe with date inputs
    df_analysis = create_analysis_dataframe(alpha_vantage_api_key, stock_abbreviation)
    if df_analysis is not None:
        print(df_analysis.head())
        break  # Break the loop if data is successfully loaded
    else:
        print("Error creating analysis dataframe. Please try again.")

def save_plot_with_choice(plt, folder_name, file_name):
    save_plot_choice = input("Do you want to save the plot? (y/n): ").lower()

    # Create a folder on the desktop
    desktop_path = os.path.join(os.path.expanduser('~'), 'Desktop')
    folder_path = os.path.join(desktop_path, folder_name)

    # Check if the folder exists, if not, create it
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    if save_plot_choice == 'y':
        # Save the plot inside the folder
        plot_path = os.path.join(folder_path, file_name)
        plt.savefig(plot_path)
        print(f'Plot has been saved in the folder "{folder_name}" as "{file_name}".')

def plot_price_trends(df_analysis, user_input):
    # Set the figure size
    plt.figure(figsize=(18, 9))

    # Mapping of user input to column names
    column_mapping = {'1': 'Open', '2': 'High', '3': 'Low', '4': 'Close'}

    # Extract column names based on user input
    selected_columns = [column_mapping[column] for column in user_input]

    try:
        # Plotting selected columns
        for column in selected_columns:
            plt.plot(df_analysis.index, df_analysis[column], label=column)

        # Adding labels and title
        plt.xlabel('Date')
        plt.ylabel('Price in USD')
        plt.title('Stock Price Trends')
        plt.legend()

        save_plot_with_choice(plt, 'Analysis_Results', f'Stock_Price_Trends_{user_input}.png')
        plt.show()

    except KeyError:
        print("Error: Invalid column name. Please enter valid column numbers.")
        user_menu()  # Loop back to the last input
        return

def volume_bar(df_analysis):
    # Create a bar chart for trading volume
    plt.figure(figsize=(18, 9))
    plt.bar(df_analysis.index,df_analysis['Volume'],color='cyan')

    # Adding labels and  title
    plt.xlabel('Date')
    plt.ylabel('Volume in USD')
    plt.title('Trading Volume')

    save_plot_with_choice(plt, 'Analysis_Results', 'Volume_Bar_Chart.png')
    plt.show()

def moving_average(df_analysis, column_name, window_size):
    # Check if the selected column exists in the dataframe
    df_analysis = df_analysis.bfill()
    valid_columns = ['Open', 'High', 'Low', 'Close', 'Volume']

    if column_name not in valid_columns:
        print(f"Error: Invalid column name. Please enter a valid column name.")
        return

    while True:
        try:
            # Check if window size is greater than the length of the DataFrame
            if window_size > len(df_analysis):
                raise ValueError("Error: Moving average window size is greater than the length of the data.")

            # Calculating moving averages
            ma_column_name = f'{column_name}_{window_size}_Day_MA'
            df_analysis.loc[:, ma_column_name] = df_analysis[column_name].rolling(window=window_size).mean()

            # Plotting the moving averages
            plt.figure(figsize=(18, 9))
            plt.plot(df_analysis.index, df_analysis[column_name], label=f'{column_name}', color='red')
            plt.plot(df_analysis.index, df_analysis[ma_column_name], label=f'{window_size}-Day MA', color='green')
            plt.xlabel('Date')
            plt.ylabel('Price/Volume')
            plt.title(f'Moving Averages ({window_size} Days) for {column_name}')
            plt.legend()

            save_plot_with_choice(plt, 'Analysis_Results', 'Moving_Average_Plot.png')
            plt.show()

            break  # Exit the loop if successful

        except ValueError as e:
            print(f"Error: {e}")
            window_size = int(input('Enter the number of days for the moving average: '))

    return

def volatility(df_analysis, window_size_input):
    while True:
        try:
            df_analysis.loc[:, 'Daily Returns'] = df_analysis['Close'].pct_change()

            if window_size_input > len(df_analysis):
                raise ValueError("Error: Rolling window size is greater than the length of the data.")

            # Plotting volatility over time
            plt.figure(figsize=(18, 9))
            plt.plot(df_analysis.index, df_analysis['Daily Returns'].rolling(window=window_size_input).std(),
                     label=f'Volatility ({window_size_input}-Day Rolling Std)', color='blue')
            plt.xlabel('Date')
            plt.ylabel('Volatility')
            plt.title('Volatility Trends')
            plt.legend()

            volatility = df_analysis['Daily Returns'].std()
            print(f'Standard deviation of daily returns as a measure of volatility: {volatility}')

            save_plot_with_choice(plt, 'Analysis_Results', 'Daily_Returns_Plot.png')
            plt.show()

            break  # Exit the loop if successful

        except ValueError as e:
            print(f"Error: {e}")
            window_size_input = int(input('Enter the number of days for the volatility calculation: '))

    return

def correlation(df_analysis):
    # Calculating correlation matrix
    correlation_matrix = df_analysis[['Open', 'High', 'Low', 'Close', 'Volume']].corr()

    # Plotting a heatmap of the correlation matrix
    plt.figure(figsize=(16,12))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap')

    save_plot_with_choice(plt, 'Analysis_Results', 'Correlation_Heatmap.png')
    plt.show()

def relationship(df_analysis):
    print("Pairplot options:")
    print("1. All ('Open', 'High', 'Low', 'Close', 'Volume')")
    print("2. Custom")

    pairplot_choice = input("Enter your selection: ")

    if pairplot_choice == '1':
        # Using pairplot for visualizing pairwise relationship
        sns.pairplot(df_analysis[['Open', 'High', 'Low', 'Close', 'Volume']])
        plt.title('Pair Plot of Stock Indicators')
        save_plot_with_choice(plt, 'Analysis_Results', 'Relationship_Pairplot.png')
        plt.show()
    elif pairplot_choice == '2':
        print("Enter two numbers which correspond to the relationships you want to plot:")
        print("1. Open, 2. High, 3. Low, 4. Close, 5. Volume")

        user_input = input("Enter your selection (e.g., 15): ")

        try:
            selected_columns = [df_analysis.columns[int(i) - 1] for i in user_input]
        except (ValueError, IndexError):
            print("Error: Invalid input. Please enter valid numbers.")
            relationship(df_analysis)  # Loop back to the pairplot choice
            return

        # Using pairplot for visualizing pairwise relationship for selected columns
        sns.pairplot(df_analysis[selected_columns])
        plt.title('Pair Plot of Stock Indicators')
        save_plot_with_choice(plt, 'Analysis_Results', 'Relationship_Pairplot.png')
        plt.show()
    else:
        print("Error: Invalid choice. Please enter 1 or 2.")
        relationship(df_analysis)
def user_menu():
    print("Which operation do you want to perform with your data?")
    print("1. Plot price trends (Open, High, Low, Close)")
    print("2. Volume amount")
    print("3. Moving average")
    print("4. Volatility")
    print("5. Correlation heatmap")
    print("6. Relationship pairplot")
    print("7. Prediction (10 days)")
    print("Press Enter to quit and run the program again")

    choice = input("Enter the number corresponding to your choice: ")

    if choice == '1':
        while True:
            # Ask user for input on which columns to plot
            print("Please enter the columns that you want to plot price trends of:")
            print("1. Open, 2. High, 3. Low, 4. Close")
            print("Example usage: 12 (Plot the prices of Open and High in one plot)")

            user_input = input("Enter your selection: ")

            # Validate user input and plot price trends
            if all(char.isdigit() and 1 <= int(char) <= 4 for char in user_input) and len(user_input) >= 1:
                plot_price_trends(df_analysis, user_input)
                break  # Exit the loop if input is valid
            else:
                print("Error: Invalid input. Please enter valid column numbers.")
        user_menu()
    elif choice == '2':
        volume_bar(df_analysis)
        user_menu()
    elif choice == '3':
        column_input = input("Enter the column to apply moving average on (Open/Close/High/Low/Volume): ")
        window_size_input = int(input('Enter the number of days for the moving average: '))
        moving_average(df_analysis, column_input, window_size_input)
        user_menu()

    elif choice == '4':
        window_size_input = int(input('Enter the number of days for the volatility calculation: '))
        volatility(df_analysis, window_size_input)
        user_menu()
    elif choice == '5':
        while True:
            try:
                print("Correlation options:")
                print("1. Correlation heatmap for all columns")
                print("2. Correlation between High and Low Prices")
                print("3. Correlation between Closing Price and Volume")
                print("4. Correlation between Opening Price and Closing Price")

                correlation_choice = input("Enter the number corresponding to your correlation choice: ")

                if correlation_choice == '1':
                    correlation(df_analysis)
                elif correlation_choice == '2':
                    correlation_high_low = df_analysis['High'].corr(df_analysis['Low'])
                    print(f'Correlation between High and Low Prices: {correlation_high_low}')
                elif correlation_choice == '3':
                    correlation_close_volume = df_analysis['Close'].corr(df_analysis['Volume'])
                    print(f'Correlation between Closing Price and Volume: {correlation_close_volume}')
                elif correlation_choice == '4':
                    correlation_open_close = df_analysis['Open'].corr(df_analysis['Close'])
                    print(f'Correlation between Opening Price and Closing Price: {correlation_open_close}')
                else:
                    print("Invalid correlation choice. Please enter a valid number.")
                    continue  # Loop back to the last input if an error occurs

                break  # Exit the loop if correlation calculation is successful
            except ValueError as e:
                print(f"Error: {e}")

        user_menu()  # Loop back to the main menu
        return
    elif choice == '6':
        relationship(df_analysis)
        user_menu()
    elif choice == '7':
        df_stationary = evaluate_stationarity(df['Close'])
        print(df_stationary.head())
        best_p, best_q = find_best_arima_parameters_and_forecast(df_stationary)
        forecast_and_plot(best_p, best_q,df_stationary)
        user_menu()
    else:
        exit()

user_menu()