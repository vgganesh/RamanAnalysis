import pandas as pd
import glob

def read_raman_data(file_pattern, rows_per_spectrum=391, measurement_duration=62):
    """
    Reads Raman spectroscopy data from text files matching the given pattern.

    Parameters:
        file_pattern (str): The file name pattern to match (e.g., "HS006_400_20min_60x1_1500stat_LP5_*.txt").
        rows_per_spectrum (int): Number of rows per spectrum in the data files.
        measurement_duration (int): Duration of the measurement in seconds.

    Returns:
        tuple: A tuple containing:
            - rmn (list): A list of DataFrames, each representing a spectrum.
            - selected_data (dict): A dictionary with file and spectrum identifiers as keys and spectra as values.
            - measdur (int): The measurement duration in seconds.
    """

    # Initialize an empty list for reading data and an empty dictionary for selected data
    rmn = []
    selected_data = {}

    # Get the list of files matching the pattern and sort them
    files = glob.glob(file_pattern)
    files.sort(key=str.lower)

    # Loop through the files and read them
    for n in files:
        try:
            # Read the entire file into a DataFrame
            data = pd.read_csv(n, sep='\t', header=None, names=['Depth', 'Wavenumber', 'Intensity'], skiprows=0, index_col=False)
            
            # Determine the number of spectra in the file
            num_spectra = len(data) // rows_per_spectrum
            
            # Split the data into individual spectra
            spectra = [data.iloc[i*rows_per_spectrum:(i+1)*rows_per_spectrum].drop(columns=['Depth']) for i in range(num_spectra)]
            
            # Append all spectra to the rmn list
            rmn.extend(spectra)
            
            # Store each spectrum in the selected_data dictionary with a unique key
            for i, spectrum in enumerate(spectra):
                selected_data[f"{n}_spectrum_{i+1}"] = spectrum
        except pd.errors.ParserError:
            # Handle the exception if there is a parsing error
            print(f"Error parsing file: {n}")

    return rmn, selected_data, measurement_duration