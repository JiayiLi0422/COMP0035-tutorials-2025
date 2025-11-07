import os
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Get data directory
try:
    current_file = Path(__file__)
    data_dir = current_file.parent.parent.joinpath("data")
except NameError:
    data_dir = Path(os.getcwd()).joinpath("data")

def load_datasets():
    """Load all weather datasets"""
    try:
        annual_df = pd.read_csv(data_dir / "AirTemperatureAndSunshineRelativeHumidityAndRainfallAnnual.csv", na_values=['na', ' na '])
        monthly_df = pd.read_csv(data_dir / "AirTemperatureAndSunshineRelativeHumidityAndRainfallMonthly.csv", na_values=['na', ' na '])
        sunshine_df = pd.read_csv(data_dir / "SunshineDurationMonthlyMeanDailyDuration.csv", na_values=['na', ' na '])
        return annual_df, monthly_df, sunshine_df
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        print(f"Looking for data in: {data_dir}")
        return None, None, None

def describe_dataframe(df, df_name="DataFrame"):
    """Enhanced function to describe a dataframe"""
    if df is None:
        print(f"--- {df_name} ---")
        print("Dataset not available")
        return
    
    print(f"\n{'='*50}")
    print(f"--- {df_name} ---")
    print(f"{'='*50}")
    
    # Basic info
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Data types
    print(f"\nData Types:")
    print(df.dtypes)
    
    # Missing values
    missing_values = df.isnull().sum()
    if missing_values.sum() > 0:
        print(f"\nMissing Values:")
        print(missing_values[missing_values > 0])
    else:
        print(f"\nNo missing values found")
    
    # First and last few rows
    print(f"\nFirst 5 rows:")
    print(df.head())
    
    print(f"\nLast 5 rows:")
    print(df.tail())
    
    # Basic statistics
    print(f"\nBasic Statistics:")
    print(df.describe())
    
    # Memory usage
    print(f"\nMemory Usage:")
    df.info(memory_usage='deep')

def generate_summary_statistics(df, df_name="DataFrame"):
    """Generate summary statistics for all numerical variables"""
    if df is None:
        return
    
    print(f"\n{'='*60}")
    print(f"SUMMARY STATISTICS - {df_name}")
    print(f"{'='*60}")
    
    # CRITICAL FIX: Handle wide-format data properly
    if "Annual" in df_name or "Monthly" in df_name:
        # First transpose to make variables (DataSeries) become columns
        df_transposed = df.set_index('DataSeries').T
        # Select only numeric columns from transposed data
        numeric_data = df_transposed.select_dtypes(include=[np.number])
        
        if len(numeric_data.columns) == 0:
            print("No numerical variables found after transposition.")
            return
        
        print(f"Found {len(numeric_data.columns)} numerical variables")
        print(f"Variables: {list(numeric_data.columns)}")
        
        # For datasets with many time periods, focus on recent years
        if len(numeric_data) > 20:
            recent_data = numeric_data.tail(10)  # Last 10 time periods
            print(f"\nFocusing on recent time periods (showing last 10):")
            stats_df = recent_data.describe()
        else:
            stats_df = numeric_data.describe()
        
        print(f"\nDetailed Summary Statistics:")
        print(stats_df)
        
        # Additional statistics for each variable
        print(f"\nAdditional Statistics:")
        for col in stats_df.columns:
            values = numeric_data[col].dropna()
            if len(values) > 0:
                print(f"\n{col}:")
                print(f"  Range: {values.min():.2f} to {values.max():.2f}")
                print(f"  Variance: {values.var():.2f}")
                print(f"  Skewness: {values.skew():.2f}")
    
    else:
        # For already tidy data (like Sunshine), process normally
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        if len(numeric_cols) == 0:
            print("No numerical variables found.")
            return
        
        print(f"Found {len(numeric_cols)} numerical variables")
        stats_df = df[numeric_cols].describe()
        
        print(f"\nDetailed Summary Statistics:")
        print(stats_df)
        
        # Additional statistics
        print(f"\nAdditional Statistics:")
        for col in stats_df.columns:
            values = df[col].dropna()
            if len(values) > 0:
                print(f"\n{col}:")
                print(f"  Range: {values.min():.2f} to {values.max():.2f}")
                print(f"  Variance: {values.var():.2f}")
                print(f"  Skewness: {values.skew():.2f}")

def create_visualizations(df, df_name="DataFrame"):
    """Create visualizations to understand data distributions"""
    if df is None:
        return
    
    print(f"\n{'='*60}")
    print(f"DATA VISUALIZATIONS - {df_name}")
    print(f"{'='*60}")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) == 0:
        print("No numerical variables found for visualization.")
        return
    
    # Handle different dataset structures
    if 'mean_sunshine_hrs' in df.columns:
        # Sunshine dataset - simple structure
        create_sunshine_plots(df, df_name)
    elif len(numeric_cols) > 10:
        # Weather datasets with many year columns
        create_weather_plots(df, df_name, numeric_cols)
    else:
        # Default case
        create_basic_plots(df, df_name, numeric_cols)

def create_sunshine_plots(df, df_name):
    """Keep the useful sunshine distribution plots"""
    print("Creating sunshine distribution plots...")
    
    # Combined histogram and boxplot (this is actually useful)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    df['mean_sunshine_hrs'].plot(kind='hist', bins=20, alpha=0.7, color='orange')
    plt.title('Distribution of Sunshine Hours')
    plt.xlabel('Mean Sunshine Hours')
    plt.ylabel('Frequency')
    
    plt.subplot(1, 2, 2)
    df['mean_sunshine_hrs'].plot(kind='box')
    plt.title('Sunshine Hours Box Plot')
    plt.ylabel('Mean Sunshine Hours')
    
    plt.tight_layout()
    plt.savefig('plot1_sunshine_distribution.png')
    plt.close()

def create_weather_plots(df, df_name, numeric_cols):
    """Create 3 meaningful plots for assessment"""
    print(f"Creating assessment plots for {df_name}...")
    
    if 'Annual' in df_name:
        # Plot 2: Data Quality - Missing Data Pattern
        df_processed = df.set_index('DataSeries').T
        missing_by_decade = {}
        
        for col in df_processed.columns:
            years = pd.to_numeric(df_processed.index, errors='coerce')
            valid_years = years.dropna()
            if len(valid_years) > 0:
                missing_count = df_processed[col].isna().sum()
                total_years = len(df_processed)
                missing_pct = (missing_count / total_years) * 100
                missing_by_decade[col] = missing_pct
        
        if missing_by_decade:
            plt.figure(figsize=(12, 6))
            missing_series = pd.Series(missing_by_decade)
            missing_series[missing_series > 0].plot(kind='bar')
            plt.title('Data Quality: Missing Value Percentage by Variable')
            plt.xlabel('Weather Variables')
            plt.ylabel('Missing Percentage (%)')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig('plot2_data_quality.png')
            plt.close()
        
        # Plot 3: Scale Comparison - Variable Ranges
        numeric_data = df_processed.select_dtypes(include=[np.number])
        if len(numeric_data.columns) > 0:
            # Calculate ranges for each variable
            ranges = numeric_data.max() - numeric_data.min()
            ranges = ranges.dropna()
            
            plt.figure(figsize=(14, 8))
            ax = ranges.plot(kind='bar', logy=True)
            plt.title('Variable Scale Comparison (Range Analysis)')
            plt.xlabel('Weather Variables')
            plt.ylabel('Range (Log Scale)')
            plt.xticks(rotation=45, ha='right')  # ha='right' aligns labels properly
            plt.tight_layout()
            plt.savefig('plot3_scale_comparison.png', bbox_inches='tight')  # bbox_inches='tight' prevents label cutoff
            plt.close()

def create_basic_plots(df, df_name, numeric_cols):
    """No additional plots needed"""
    pass

def analyze_datasets():
    """Main analysis function"""
    print("Loading weather datasets...")
    annual_df, monthly_df, sunshine_df = load_datasets()
    
    # Describe each dataset
    describe_dataframe(annual_df, "Annual Temperature & Weather Data")
    describe_dataframe(monthly_df, "Monthly Temperature & Weather Data")
    describe_dataframe(sunshine_df, "Monthly Sunshine Duration Data")
    
    # Generate summary statistics for all numerical variables
    generate_summary_statistics(annual_df, "Annual Weather Data")
    generate_summary_statistics(monthly_df, "Monthly Weather Data")
    generate_summary_statistics(sunshine_df, "Sunshine Duration Data")
    
    # Create visualizations to understand data distributions
    create_visualizations(sunshine_df, "Sunshine Duration Data")
    create_visualizations(annual_df, "Annual Weather Data")
    create_visualizations(monthly_df, "Monthly Weather Data")

if __name__ == "__main__":
    analyze_datasets()