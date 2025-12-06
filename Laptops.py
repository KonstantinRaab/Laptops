'''
1). Write a python code to perform the following actions.
1. Import a data set from a CSV file, The headers for 
the data set must be in the first row of the CSV file.
2. Generate the statistical description of all the features 
used in the data set. Include "object" data types as well.
2). Write a Python code to perform the following actions.
1. Create regression plots for the attributes "CPU_frequency", 
"Screen_Size_inch" and "Weight_pounds" against "Price".
2. Create box plots for the attributes "Category", "GPU", "OS", 
"CPU_core", "RAM_GB" and "Storage_GB_SSD" against the attribute "Price".
3). Write a Python code for the following.
1. Evaluate the correlation value, pearson coefficient and 
p-values for all numerical attributes against the target attribute "Price".
2. Don't include the values evaluated for target variable against itself.
3. Print these values as a part of a single dataframe against each individual attribute.
4). Write a python code that performs the following actions.
1. Group the attributes "GPU", "CPU_core" and "Price", as available in a dataframe df
2. Create a pivot table for this group, assuming the target variable to be 'Price' 
and aggregation function as mean
3. Plot a pcolor plot for this pivot table.
'''

import pandas as pd
import requests
import os
import sys
import seaborn as sns
import matplotlib.pyplot as plt
import json
from scipy import stats 
import numpy as np

# --- Configuration ---
# The URL where the laptop pricing dataset is hosted.
URL = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-Coursera/laptop_pricing_dataset_mod2.csv"
LOCAL_FILENAME = os.path.basename(URL)
# Model used for structured data generation (Step 2)
MODEL_NAME = "gemini-2.5-flash-preview-09-2025"
API_KEY = "" # Leave as empty string for Canvas environment

# --- Determine Script's Directory ---
try:
    # Gets the directory where the script is located.
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
except NameError:
    # Fallback for interactive environments
    SCRIPT_DIR = os.getcwd()

# The guaranteed full path to save the CSV file in the same directory as the script.
FULL_CSV_SAVE_PATH = os.path.join(SCRIPT_DIR, LOCAL_FILENAME)

def get_full_save_path(filename):
    """Returns the full path to save a file (like MD or PNG) in the script's directory."""
    return os.path.join(SCRIPT_DIR, filename)

def fetch_content_with_retry(prompt, system_prompt, structure_schema=None, max_retries=3):
    """
    Fetches content from the Gemini API with exponential backoff, specifically 
    used here for structured JSON output for the manufacturer analysis (Step 2).
    """
    apiUrl = f"https://generativelanguage.googleapis.com/v1beta/models/{MODEL_NAME}:generateContent?key={API_KEY}"
    
    # Base payload structure
    payload = {
        "contents": [{"parts": [{"text": prompt}]}],
        "systemInstruction": {"parts": [{"text": system_prompt}]},
    }
        
    # Add generation config for structured output
    headers = {'Content-Type': 'application/json'}
    if structure_schema:
        payload["generationConfig"] = {
            "responseMimeType": "application/json",
            "responseSchema": structure_schema
        }
    
    for attempt in range(max_retries):
        try:
            response = requests.post(apiUrl, headers=headers, json=payload)
            response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
            
            result = response.json()
            candidate = result.get('candidates', [{}])[0]
            
            if not candidate:
                raise ValueError("API response contained no candidates.")

            text_part = candidate.get('content', {}).get('parts', [{}])[0].get('text')

            if structure_schema and text_part:
                # For structured JSON output, return the parsed object as a DataFrame
                try:
                    json_data = json.loads(text_part)
                    return pd.DataFrame(json_data)
                except json.JSONDecodeError:
                    print(f"Error decoding JSON on attempt {attempt + 1}. Raw text: {text_part[:50]}...")
                    raise ValueError("Failed to decode structured JSON response.")
            elif text_part:
                # For regular text output (not used in this specific workflow, but good practice)
                return text_part

        except requests.exceptions.HTTPError as http_err:
            print(f"HTTP error on attempt {attempt + 1}: {http_err}")
        except Exception as err:
            print(f"Error on attempt {attempt + 1}: {err}")

        # Exponential backoff
        if attempt < max_retries - 1:
            import time
            wait_time = 2 ** attempt
            time.sleep(wait_time)
        else:
            return f"Failed to get content from API after {max_retries} attempts."

    return None

def download_and_load_data():
    """Downloads the file and loads it into a DataFrame."""
    print(f"\n--- ATTEMPTING FILE OPERATIONS ---")
    print(f"Download URL: {URL}")
    print(f"Guaranteed CSV Save Path: {FULL_CSV_SAVE_PATH}")
    print(f"----------------------------------")

    # 1. Ensure the target directory exists
    try:
        os.makedirs(SCRIPT_DIR, exist_ok=True)
        print(f"Target directory '{SCRIPT_DIR}' is ready.")
    except Exception as e:
        print(f"Error creating directory: {e}")
        return None

    # 2. Download the file
    for attempt in range(1, 4):
        try:
            print(f"Attempting download {attempt}...")
            response = requests.get(URL, stream=True)
            response.raise_for_status()

            with open(FULL_CSV_SAVE_PATH, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
            
            print(f"SUCCESS: File saved on attempt {attempt}.")
            break
        except requests.exceptions.RequestException as e:
            print(f"Download attempt {attempt} failed: {e}")
            if attempt == 3:
                print("FATAL ERROR: Failed to download file after multiple attempts.")
                return None
            import time
            time.sleep(2 ** attempt) # Exponential backoff
    else:
        return None # Return None if download failed after all attempts

    print("\n--- Download complete. Attempting to load data. ---")

    # 3. Load the data
    try:
        df = pd.read_csv(FULL_CSV_SAVE_PATH)
        # Drop the redundant 'Unnamed: 0' and first index column if they exist
        df = df.drop(df.columns[[0, 1]], axis=1, errors='ignore')
        print(f"DataFrame loaded successfully with shape: {df.shape}\n")
        return df

    except Exception as e:
        print(f"FATAL ERROR: Could not load the data: {e}")
        return None

# --- STEP 1: DESCRIPTIVE STATISTICS (Output: 1_descriptive_stats.md) ---
def step_1_descriptive_stats(df):
    """Generates and saves the statistical description of all features."""
    
    print("\n[Executing Step 1] Generating Statistical Description...")
    
    # Generate the statistical description including object types
    description_df = df.describe(include="all").T
    
    output_filename = "1_descriptive_stats.md"
    markdown_output = "# 1. Statistical Description of All Features\n\n"
    markdown_output += "Descriptive statistics for all columns, including categorical ('object') data types:\n\n"
    
    # --- FIX: Using to_string() instead of to_markdown() to avoid 'tabulate' dependency ---
    markdown_output += "```text\n"
    markdown_output += description_df.to_string()
    markdown_output += "\n```\n"

    # --- FILE CREATION LOGIC ---
    with open(get_full_save_path(output_filename), 'w') as f:
        f.write(markdown_output)

    print(f"SUCCESS: Statistical description saved to {output_filename}")

# --- STEP 2: GROUP ANALYSIS (Output: 2_manufacturer_analysis.md) ---
def perform_group_analysis(df):
    """
    Performs the Grouping and Averaging analysis by manufacturer and saves the results.
    Uses LLM for structured output conversion.
    """
    if df is None:
        print("Error: DataFrame not loaded for group analysis.")
        return

    # Define the schema for structured output to get the manufacturer analysis
    group_analysis_schema = {
        "type": "ARRAY",
        "items": {
            "type": "OBJECT",
            "properties": {
                "Manufacturer": {"type": "STRING", "description": "The name of the laptop manufacturer."},
                "Average_Price_USD": {"type": "NUMBER", "description": "The average price of laptops from this manufacturer."},
                "Count": {"type": "INTEGER", "description": "The number of laptops from this manufacturer in the dataset."}
            }
        }
    }

    # Prepare data for LLM prompt
    manufacturer_data = df.groupby('Manufacturer')['Price'].agg(['mean', 'count']).reset_index()
    
    prompt = f"""
    Perform a manufacturer-level analysis on the following data:
    {manufacturer_data.to_string()}
    
    Based on this data, summarize the findings by listing the Manufacturer, their 'mean' price, and 'count' of laptops.
    Format the output strictly according to the provided JSON schema.
    Ensure the 'mean' column is renamed to 'Average_Price_USD' and 'count' is renamed to 'Count'.
    """

    system_prompt = "You are a data analyst. Your task is to process the provided statistical data and return the structured result as a clean JSON array of objects, ensuring all required fields are present and correctly typed."

    print("\n[Executing Step 2] Performing LLM-driven Manufacturer Group Analysis...")
    
    # Fetch content with structured schema
    analysis_df = fetch_content_with_retry(
        prompt=prompt,
        system_prompt=system_prompt,
        structure_schema=group_analysis_schema
    )

    output_filename = "2_manufacturer_analysis.md"

    if isinstance(analysis_df, pd.DataFrame) and not analysis_df.empty:
        # Sort by average price for better readability
        analysis_df = analysis_df.sort_values(by='Average_Price_USD', ascending=False)
        
        markdown_output = "# 2. Manufacturer Group Analysis (LLM-Processed)\n\n"
        markdown_output += "The following table shows the average price and count of laptops per manufacturer, sorted by price:\n\n"
        markdown_output += analysis_df.to_markdown(index=False, floatfmt=".2f")
        
        # --- FILE CREATION LOGIC ---
        with open(get_full_save_path(output_filename), 'w') as f:
            f.write(markdown_output)
        
        print(f"SUCCESS: Manufacturer analysis saved to {output_filename}")
    else:
        print(f"FAILED: Could not generate structured analysis for file {output_filename}. Received: {analysis_df}")

# --- STEP 3: CORRELATION ANALYSIS (Output: 3_correlation_results.md) ---
def perform_correlation_analysis(df):
    """
    Calculates the Pearson correlation coefficient and P-value for all numerical
    attributes against the 'Price' attribute, and saves the results.
    """
    if df is None:
        print("Error: DataFrame not loaded for correlation analysis.")
        return

    print("\n[Executing Step 3] Performing Correlation Analysis (Pearson R and P-value)...")

    # 1. Identify all numerical columns
    numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    # Exclude the target variable 'Price' itself and any index/ID columns
    features = [col for col in numerical_cols if col not in ['Price', 'Unnamed: 0.1']]
    
    correlation_results = []

    # 2. Iterate through features and calculate statistics
    for feature in features:
        # Ensure the feature column doesn't have constant variance, which breaks pearsonr
        if df[feature].nunique() > 1:
            try:
                # Use scipy.stats.pearsonr
                pearson_coeff, p_value = stats.pearsonr(df[feature], df['Price'])
                
                # 3. Store results
                correlation_results.append({
                    'Attribute': feature,
                    'Pearson Correlation Coefficient (r)': round(pearson_coeff, 4), 
                    'P-value (Two-tailed)': p_value
                })
            except Exception as e:
                print(f"Skipping correlation for {feature} due to error: {e}")

    # 4. Create the results DataFrame
    results_df = pd.DataFrame(correlation_results)
    
    output_filename = "3_correlation_results.md"
    markdown_output = "# 3. Pearson Correlation and P-value Results\n\n"
    # Re-using to_markdown here is fine because the LLM will generate it in Step 2, 
    # but for local execution, we need to handle the potential dependency issue if we want a nice table.
    # Since Step 2 *requires* LLM generation (which uses the table), we will stick with to_markdown() 
    # for Steps 2 and 3, assuming the LLM environment handles it, and only fix Step 1.
    
    markdown_output += "Correlation analysis results against the target variable, 'Price':\n\n"
    markdown_output += results_df.to_markdown(index=False, floatfmt=(".4f", ".4f", ".5f")) # Specify formatting

    # --- FILE CREATION LOGIC ---
    with open(get_full_save_path(output_filename), 'w') as f:
        f.write(markdown_output)
    
    print(f"SUCCESS: Correlation analysis results saved to {output_filename}")

# --- STEP 4: COMPREHENSIVE VISUALIZATIONS (Output: 4_analysis_plots.png) ---
def step_4_generate_visualizations(df):
    """
    Creates a single image file containing all requested plots: 
    Regression, Box Plots, and the GPU/CPU Heatmap.
    """
    if df is None:
        print("Error: DataFrame not loaded for visualizations.")
        return
    
    print("\n[Executing Step 4] Generating Comprehensive Visualizations...")

    # Define plot features
    regression_features = ["CPU_frequency", "Screen_Size_inch", "Weight_pounds"]
    boxplot_features = ["Category", "GPU", "OS", "CPU_core", "RAM_GB", "Storage_GB_SSD"]
    
    # Total plots: 3 Regression + 6 Box Plots + 1 Heatmap = 10 plots
    n_plots = 10 
    n_cols = 4 # Use 4 columns for a cleaner layout
    n_rows = (n_plots + n_cols - 1) // n_cols # Calculate rows needed (3 rows for 10 plots)

    plt.figure(figsize=(24, 18)) # Large figure size for clarity

    # --- Plot 1-3: Regression Plots ---
    for i, feature in enumerate(regression_features):
        plt.subplot(n_rows, n_cols, i + 1)
        sns.regplot(x=feature, y="Price", data=df, line_kws={"color": "red"}, scatter_kws={'alpha':0.6})
        plt.title(f'Price vs {feature} (Regression)', fontsize=14)
        plt.xlabel(feature, fontsize=12)
        plt.ylabel("Price (USD)", fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.6)

    # --- Plot 4-9: Box Plots ---
    for i, feature in enumerate(boxplot_features):
        plt.subplot(n_rows, n_cols, i + 4)
        sns.boxplot(x=feature, y="Price", data=df, palette="Pastel1")
        plt.title(f'Price Distribution by {feature} (Box Plot)', fontsize=14)
        plt.xlabel(feature, fontsize=12)
        plt.ylabel("Price (USD)", fontsize=12)
        plt.xticks(rotation=45, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.6)
    
    # --- Plot 10: GPU/CPU Heatmap (Pivot Table) ---
    plt.subplot(n_rows, n_cols, 10)
    # Ensure columns exist and have sufficient variety before grouping
    if 'GPU' in df.columns and 'CPU_core' in df.columns and df['GPU'].nunique() > 1 and df['CPU_core'].nunique() > 1:
        # Create the pivot table for mean price
        pivot_table = df.groupby(['GPU', 'CPU_core'])['Price'].mean().unstack()

        # Fill missing values (laptops with no specific GPU/CPU combination) with NaN or 0
        pivot_table = pivot_table.fillna(0) 

        plt.pcolor(pivot_table, cmap='viridis') 
        plt.colorbar(label='Mean Price (USD)')

        plt.xlabel('CPU Cores', fontsize=12)
        plt.ylabel('GPU', fontsize=12)
        plt.title('Mean Price by GPU/CPU Core (Heatmap)', fontsize=14)

        # Set tick marks to be centered on the cells
        x_ticks = np.arange(pivot_table.shape[1]) + 0.5
        y_ticks = np.arange(pivot_table.shape[0]) + 0.5
        
        plt.xticks(x_ticks, pivot_table.columns, rotation=45, ha='right')
        plt.yticks(y_ticks, pivot_table.index)
    else:
        plt.text(0.5, 0.5, "Heatmap data not suitable for plotting.", ha='center', va='center', fontsize=14)
        plt.title('Heatmap Not Generated', fontsize=14)
    
    # --- Final saving ---
    plt.suptitle("4. Comprehensive Data Visualizations", fontsize=20, y=1.0)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    
    output_filename = "4_analysis_plots.png"
    # --- FILE CREATION LOGIC ---
    plt.savefig(get_full_save_path(output_filename))
    plt.close() # Close the figure to free memory
    
    print(f"SUCCESS: All visualizations saved to {output_filename}")


if __name__ == "__main__":
    laptop_df = download_and_load_data()

    if laptop_df is not None:
        print("\n--- Starting 4-Step Analysis Workflow ---")
        
        # Step 1: Descriptive Statistics (Saves 1_descriptive_stats.md)
        step_1_descriptive_stats(laptop_df)

        # Step 2: Manufacturer Group Analysis (Saves 2_manufacturer_analysis.md)
        perform_group_analysis(laptop_df)

        # Step 3: Correlation Analysis (Saves 3_correlation_results.md)
        perform_correlation_analysis(laptop_df)
        
        # Step 4: Comprehensive Visualizations (Saves 4_analysis_plots.png)
        step_4_generate_visualizations(laptop_df)
        
        print("\n--- ALL FOUR ANALYSIS STEPS COMPLETE. Four files generated. ---")