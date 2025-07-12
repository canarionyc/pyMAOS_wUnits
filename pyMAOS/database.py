import os
import csv

from pyMAOS.material import LinearElasticMaterial as Material 


# --- Read materials ---
def get_materials_from_csv(csv_file):
    """
    Read materials from CSV file using pandas for improved handling of whitespace and empty lines.
    
    Args:
        csv_file: Path to the CSV file containing material properties
        
    Returns:
        Dictionary of materials indexed by material UID
    """
    import pandas as pd
    print(f"Reading materials from {csv_file}...")
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV file not found: {csv_file}")  
    materials = {}
    try:
        # Determine the ID column name before reading
        with open(csv_file, 'r') as f:
            first_line = f.readline().strip()
            id_col = "mat_uid" if "mat_uid" in first_line else "uid"
        
        # Read CSV with pandas - handles whitespace and blank lines better
        df = pd.read_csv(
            csv_file,
            skipinitialspace=True,
            skip_blank_lines=True,
            index_col=id_col  # Use the ID column as index
        )
        
        # Clean whitespace in all string columns
        for col in df.select_dtypes(include=['object']):
            df[col] = df[col].str.strip()
        
        # Check for duplicate indices
        if len(df.index) != len(df.index.unique()):
            duplicates = df.index[df.index.duplicated()].unique()
            raise ValueError(f"Duplicate material IDs found: {duplicates}")
        
        # Create materials dictionary directly from index
        for uid, row in df.iterrows():
            materials[int(uid)] = Material(
                float(row["density"]),
                float(row["E"]),
                # Add Poisson's ratio if available, otherwise use default 0.3
                float(row["nu"]) if "nu" in row else 0.3
            )
    except Exception as e:
        print(f"Error reading materials from {csv_file}: {e}")
        raise
        
    return materials

# --- Read sections ---
from pyMAOS.section import Section
def get_sections_from_csv(csv_file):
    """
    Read sections from CSV file using pandas for improved handling of whitespace and empty lines.
    
    Args:
        csv_file: Path to the CSV file containing section properties
        
    Returns:
        Dictionary of sections indexed by section UID
    """
    import pandas as pd
    print(f"Reading sections from {csv_file}...")
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV file not found: {csv_file}")
    sections_dict = {}
    try:
        # Determine the ID column name before reading
        with open(csv_file, 'r') as f:
            first_line = f.readline().strip()
            id_col = "sect_uid" if "sect_uid" in first_line else "uid"
        
        # Read CSV with pandas
        df = pd.read_csv(
            csv_file,
            skipinitialspace=True,
            skip_blank_lines=True,
            index_col=id_col
        )
        
        # Clean whitespace in all string columns
        for col in df.select_dtypes(include=['object']):
            df[col] = df[col].str.strip()
        
        # Check for duplicate indices
        if len(df.index) != len(df.index.unique()):
            duplicates = df.index[df.index.duplicated()].unique()
            raise ValueError(f"Duplicate section IDs found: {duplicates}")
        
        # Create sections dictionary directly from index
        for uid, row in df.iterrows():
            sections_dict[int(uid)] = Section(
                float(row["Area"]), 
                float(row["Ixx"]), 
                float(row["Iyy"])
            )
        
        return sections_dict
        
    except Exception as e:
        print(f"Error reading sections from {csv_file}: {e}")
        raise
