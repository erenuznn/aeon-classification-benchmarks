import pandas as pd  # Data manipulation library used to read the processed CSV files
import numpy as np  # Numerical library used for high-speed array reshaping and math
import os  # Operating system interface for path validation and directory creation


def generate_unified_unilateral_ts(file_label_mapping, output_path, dataset_name, points_per_window=3600):
    """
    Ingests multiple processed CSV files and outputs a single aggregated unilateral .ts file.

    Parameters:
    - file_label_mapping: Dictionary linking file paths (keys) to Stress/Adequate labels (values).
    - output_path: Where the final .ts file will be saved on your MacBook.
    - dataset_name: The name used in the metadata header (e.g., Vivent_Master_Unilateral).
    - points_per_window: The temporal length of one instance (3600 seconds/1 hour).
    """

    all_instances = []          # Initialize an empty list to store the formatted text rows for every hour
    total_processed_files = 0   # Counter to track how many of the 24 plants were successfully loaded

    # Begin iterating through the plant-to-label dictionary provided by the Master Pipeline
    for csv_path, state_label in file_label_mapping.items():
        if not os.path.exists(csv_path):                             # Check if processed csv file exists
            print(f"Warning: File not found {csv_path}. Skipping.")  # Log the missing file
            continue                                                 # Move to the next plant in the dictionary

        # --- Data Loading ---
        df = pd.read_csv(csv_path)           # Load the filtered CSV into a DataFrame
        signal_array = df.iloc[:, 1].values  # Extract only the signal amplitude as a NumPy array

        # --- Segmentation ---
        # Calculate how many full 1-hour "chunks" exist in this specific plant's data
        total_valid_chunks = len(signal_array) // points_per_window

        if total_valid_chunks > 0:                                                            # Only proceed if there is at least 1 hour of data
            truncated_signal = signal_array[:total_valid_chunks * points_per_window]          # Trim off any "leftover" seconds at the end that don't make a full hour
            feature_matrix = truncated_signal.reshape(total_valid_chunks, points_per_window)  # RESHAPE: Turn a flat list of points into a matrix where each row is 1 hour (3600 points)

            # --- Format rows and append to master list ---
            for row in feature_matrix:                                  # Process each 1-hour window individually
                row_str = ",".join([f"{val:.6f}" for val in row])       # Convert the 3600 floats into a single string separated by commas, rounded to 6 decimals
                all_instances.append(f"{row_str}:{state_label}\n")      # Combine the signal string with the Stress Label (e.g., "0.1,0.2...:1")
            total_processed_files += 1                                  # Increment the successfully processed plant count

    if not all_instances:  # If no data was processed (e.g., all files missing)
        raise ValueError("Execution halted: No valid instances generated from provided files.")

    # --- File Generation ---
    os.makedirs(os.path.dirname(output_path), exist_ok=True)    # Create the 'processed_ts' directory if it doesn't already exist on your drive

    with open(output_path, 'w') as file:              # Open the target .ts file for writing
        # --- Write Metadata Header Block ---
        file.write(f"@problemName {dataset_name}\n")  # Write the experimental identifier
        file.write("@timeStamps false\n")             # Tell the loader that time info is implicit (1Hz), not explicit
        file.write("@missing false\n")                # Confirm there are no 'NaN' or empty values in the signal
        file.write("@univariate true\n")              # Declares this is a unilateral (1-sensor) dataset
        file.write("@equalLength true\n")             # Confirms every single row is exactly 3600 points long

        # --- Extract unique labels dynamically for header ---
        unique_labels = sorted(list(set(file_label_mapping.values())))   # Find all unique labels (0 and 1) used in the dictionary to inform the loader
        label_str = " ".join(map(str, unique_labels))                    # Create a string like "0 1"
        file.write(f"@classLabel true {label_str}\n")                    # Define the classification categories

        file.write("@data\n")                                            #  End of metadata and start of raw data

        # --- Write Data Matrix ---
        for instance in all_instances:  # Loop through the formatted strings
            file.write(instance)        # Write the full hour of data and its label to the file

    print(f"Unified unilateral .ts transformation complete.")
    print(f"Processed {total_processed_files} files.")
    print(f"Total aggregated instances: {len(all_instances)}.")

    return output_path  # Return the path to the newly created .ts file