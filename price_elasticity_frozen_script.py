import argparse
import pandas as pd
from scipy.stats import linregress
import warnings
import numpy as np
import sys
import os


# TO RUN : python -m  price_elasticity_frozen_script "C:\Main\cs\plutoconsultingservices\elasticityanalysis\Price_Elasticity_Frozen_Initial.csv" -o "C:\Main\cs\plutoconsultingservices\elasticityanalysis\Price_Elasticity_Test_Executable.csv"
# CONVERT TO EXE : pyinstaller --onefile --windowed price_elasticity_frozen_script.py
# TO RUN EXE : cd "C:\Main\cs\plutoconsultingservices\elasticityanalysis\dist" || THEN RUN : .\price_elasticity_frozen_script.exe "C:\Main\cs\plutoconsultingservices\elasticityanalysis\Price_Elasticity_Frozen_Initial.csv" -o "C:\Main\cs\plutoconsultingservices\elasticityanalysis\Price_Elasticity_Test_Executable.csv"
def safe_linregress(x, y):
    """
    Calculate linear regression parameters with exception handling for edge cases.
    
    Returns a tuple of (slope, intercept, r_squared) with appropriate defaults for error cases.
    """
    try:
        # Check if we have enough unique x values for regression
        if len(np.unique(x)) < 2:
            # Not enough unique points for regression
            return (np.inf, 0, 0)
        
        # Perform linear regression
        result = linregress(x, y)
        return (result.slope, result.intercept, result.rvalue**2)
    
    except (ValueError, TypeError, ZeroDivisionError) as e:
        # This captures cases where:
        # - Slope is infinite (perfect vertical line)
        # - Invalid input data
        # - Division by zero errors
        warnings.warn(f"Linear regression failed: {str(e)}. Returning default values.")
        return (0, 0, 0)  # Default values for error cases

# Correct approach to calculate mean of two values
def safe_mean(a, b):
    """Calculate mean of two values safely, handling different data types"""
    # First convert both inputs to numeric values
    try:
        a_numeric = pd.to_numeric(a, errors='coerce')
        b_numeric = pd.to_numeric(b, errors='coerce')
        return (a_numeric + b_numeric) / 2
    except Exception as e:
        warnings.warn(f"Error in safe_mean: {str(e)}")
        return np.nan

# Wrapper function for division to handle edge cases
def safe_division(numerator, denominator):
    """Perform division safely, handling zeros and type mismatches"""
    try:
        # Convert inputs to numeric values
        num = pd.to_numeric(numerator, errors='coerce') 
        den = pd.to_numeric(denominator, errors='coerce')
        
        # Replace zeros in denominator to avoid division by zero
        den = np.where(np.isclose(den, 0), np.nan, den)
        
        # Perform division
        result = num / den
        
        # Replace infinities with NaN
        result = np.where(np.isinf(result), np.nan, result)
        
        return result
    except Exception as e:
        warnings.warn(f"Error in safe_division: {str(e)}")
        return np.nan


def process_data(input_file, output_file, **kwargs):
    """
    Process the input CSV file and save results to the output CSV.
    Add your notebook's data analysis code here.
    
    Parameters:
    -----------
    input_file : str
        Path to input CSV file
    output_file : str
        Path to save the output CSV file
    **kwargs : 
        Additional parameters for data processing
    """
    try:
        print(f"Reading data from {input_file}...")
        
        # Read the input CSV
        df = pd.read_csv(input_file, delimiter=kwargs.get('delimiter', ','), 
                encoding=kwargs.get('encoding', 'utf-8'))
        
        print(f"Processing data... (Found {len(df)} rows)")
        
        #dftemp = df.progress_apply(results)
        
        # =============================================
        # ADD YOUR NOTEBOOK'S ANALYSIS CODE HERE
        
        df_main = df.groupby(["Item"]).agg(
            {
                "Unit_Retail": ["mean", "sum", "min", "max", np.median],
                "Total_units": ["mean", "sum", "min", "max", np.median],
                "Item": ["count"]
            }
        )
        
        results = df.groupby(["Item"])["Unit_Retail"].apply(
            lambda x: safe_linregress(x.values, df.loc[x.index, "Total_units"].values)
        )
                
        df_main["Slope"] = results.apply(lambda x: x[0])
        df_main["Intercept"] = results.apply(lambda x: x[1])
        df_main["R_squared"] = results.apply(lambda x: x[2])
        
        df_main["predicted_min_quantity"] = df_main["Intercept"] + df_main["Slope"] * df_main["Unit_Retail"]["min"]
        df_main["predicted_max_quantity"] = df_main["Intercept"] + df_main["Slope"] * df_main["Unit_Retail"]["max"]
        df_main["predicted_median_quantity"] = df_main["Intercept"] + df_main["Slope"] * df_main["Unit_Retail"]["median"]
        df_main["predicted_average_quantity"] = df_main["Intercept"] + df_main["Slope"] * df_main["Unit_Retail"]["mean"]
        
        # =============================================
        
        
        
        # =============================================
        
        try:
            # For min_avg_elasticity
            # Get price values, ensuring they're numeric
            price_mean = pd.to_numeric(df_main["Unit_Retail"]["mean"], errors='coerce')
            price_min = pd.to_numeric(df_main["Unit_Retail"]["min"], errors='coerce')
            
            # Calculate quantity ratio (first part of elasticity formula)
            quantity_change_min = df_main["predicted_average_quantity"] - df_main["predicted_min_quantity"]
            quantity_avg_min = safe_mean(df_main["predicted_average_quantity"], df_main["predicted_min_quantity"])
            quantity_ratio_min = safe_division(quantity_change_min, quantity_avg_min)
            
            # Calculate price ratio (second part of elasticity formula)
            price_change_min = price_mean - price_min
            price_avg_min = safe_mean(price_mean, price_min)
            price_ratio_min = safe_division(price_change_min, price_avg_min)
            
            # Calculate elasticity
            df_main["min_avg_elasticity"] = safe_division(quantity_ratio_min, price_ratio_min)
            
            # For max_avg_elasticity
            # Get price values, ensuring they're numeric
            price_max = pd.to_numeric(df_main["Unit_Retail"]["max"], errors='coerce')
            
            # Calculate quantity ratio (first part of elasticity formula)
            quantity_change_max = df_main["predicted_max_quantity"] - df_main["predicted_average_quantity"]
            quantity_avg_max = safe_mean(df_main["predicted_max_quantity"], df_main["predicted_average_quantity"])
            quantity_ratio_max = safe_division(quantity_change_max, quantity_avg_max)
            
            # Calculate price ratio (second part of elasticity formula)
            price_change_max = price_max - price_mean
            price_avg_max = safe_mean(price_max, price_mean)
            price_ratio_max = safe_division(price_change_max, price_avg_max)
            
            # Calculate elasticity
            df_main["max_avg_elasticity"] = safe_division(quantity_ratio_max, price_ratio_max)
            
            # Calculate average elasticity
            df_main["avg_elasticity"] = safe_mean(
                pd.to_numeric(df_main["min_avg_elasticity"], errors='coerce'),
                pd.to_numeric(df_main["max_avg_elasticity"], errors='coerce')
            )
            
            
            # Flag calculations that might not be valid (using numeric comparison)
            '''df_main["valid_elasticity"] = (
                ~df_main["min_avg_elasticity"].isna() & 
                ~df_main["max_avg_elasticity"].isna() &
                (np.abs(df_main["min_avg_elasticity"]) < 1000) &
                (np.abs(df_main["max_avg_elasticity"]) < 1000)
            )'''
            
            # Optional: Fill NaNs with 0 or other appropriate value
            for col in ["min_avg_elasticity", "max_avg_elasticity", "avg_elasticity"]:
                invalid_count = df_main[col].isna().sum()
                if invalid_count > 0:
                    print(f"Found {invalid_count} invalid values in {col}. Replacing with 0.")
                    df_main[col].fillna(0, inplace=True)
                    
            print("Elasticity calculations completed successfully.")

        except Exception as e:
            print(f"Error calculating elasticities: {str(e)}")
            # Initialize columns with zeros
            df_main["min_avg_elasticity"] = 0
            df_main["max_avg_elasticity"] = 0
            df_main["avg_elasticity"] = 0
            #df_main["valid_elasticity"] = False
        
        df_main = df_main.reset_index()
        
        # =============================================
        
        # Example transformation (replace with your actual analysis):
        '''if kwargs.get('normalize', False):
            numeric_columns = df.select_dtypes(include=['number']).columns
            df[numeric_columns] = df[numeric_columns].apply(
                lambda x: (x - x.min()) / (x.max() - x.min()) if x.max() > x.min() else x
            )
            
        # Optional filtering based on threshold
        if 'threshold' in kwargs:
            column = kwargs.get('filter_column')
            if column and column in df.columns:
                df = df[df[column] > kwargs['threshold']]'''
        
        # =============================================
        
        # Save the result
        df_main.to_csv(output_file, index=kwargs.get('include_index', True))
        print(f"Output saved to {output_file}")
        return True
        
    except Exception as e:
        print(f"Error: {str(e)}", file=sys.stderr)
        return False

def main():
    # Set up command-line argument parsing
    parser = argparse.ArgumentParser(
        description='Process a CSV file and output the results to another CSV file.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    
    
    # Required arguments
    parser.add_argument('input', help='Path to the input CSV file')
    parser.add_argument('-o', '--output', help='Path to the output CSV file')
    parser.add_argument('--delimiter', default=',', help='CSV delimiter character')
    parser.add_argument('--encoding', default='utf-8', help='File encoding')
    
    # Optional arguments - customize based on your notebook's needs
    
    parser.add_argument('--normalize', action='store_true', help='Normalize numeric columns')
    parser.add_argument('--filter-column', help='Column name to use for threshold filtering')
    parser.add_argument('--threshold', type=float, help='Threshold value for filtering')
    parser.add_argument('--include-index', action='store_true', help='Include index in output CSV')
    
    args = parser.parse_args()
    
    # If output file is not specified, create it from the input filename
    if not args.output:
        input_base = os.path.splitext(args.input)[0]
        args.output = f"{input_base}_processed.csv"
    
    # Call the processing function with the provided arguments
    success = process_data(
        args.input, 
        args.output,
        normalize=args.normalize,
        filter_column=args.filter_column,
        threshold=args.threshold,
        include_index=args.include_index
    )
    
    # Return appropriate exit code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
