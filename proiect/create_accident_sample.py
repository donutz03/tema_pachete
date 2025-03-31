import pandas as pd
import os


def extract_sample_data(input_file, output_file, samples_per_year=1000):
    """
    Extract the first 1000 records from each year from the US_Accidents dataset.

    Parameters:
    -----------
    input_file : str
        Path to the original CSV file
    output_file : str
        Path where the sampled CSV will be saved
    samples_per_year : int
        Number of samples to extract per year (default: 1000)
    """
    print(f"Reading original dataset from {input_file}...")
    # Read the original dataset
    df = pd.read_csv(input_file)

    # Convert Start_Time to datetime with format='mixed' to handle different datetime formats
    df['Start_Time'] = pd.to_datetime(df['Start_Time'], format='mixed')

    # Extract year from Start_Time
    df['Year'] = df['Start_Time'].dt.year

    # Get unique years in the dataset
    years = df['Year'].unique()
    print(f"Found years in the dataset: {sorted(years)}")

    # Create empty dataframe to hold the samples
    sampled_data = pd.DataFrame()

    # Extract samples per year
    for year in sorted(years):
        # Get data for the current year
        year_data = df[df['Year'] == year]

        # Take the first 'samples_per_year' records
        year_sample = year_data.head(samples_per_year)

        # Add to the sampled data
        sampled_data = pd.concat([sampled_data, year_sample])

        print(f"Extracted {len(year_sample)} records from year {year}")

    # Drop the Year column we added
    sampled_data = sampled_data.drop('Year', axis=1)

    # Save to CSV
    sampled_data.to_csv(output_file, index=False)
    print(f"Sampled dataset saved to {output_file}")
    print(f"Total records in sampled dataset: {len(sampled_data)}")


if __name__ == "__main__":
    # Define file paths
    input_file = "US_Accidents_March23.csv"
    output_file = "US_Accidents_Sample_1000_Per_Year.csv"

    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        exit(1)

    # Extract sample data
    extract_sample_data(input_file, output_file)