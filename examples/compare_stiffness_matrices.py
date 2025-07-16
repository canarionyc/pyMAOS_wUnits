import os
import numpy as np
import pandas as pd
import glob
from pathlib import Path
import matplotlib.pyplot as plt

def compare_stiffness_matrices():
    # Directories to compare
    orig_dir = "braced_frame_output_ORIG"
    new_dir = "braced_frame_output"
    
    # Pattern to match
    pattern = "member_*_global_stiffness.csv"
    
    # Find all member files in original directory
    orig_files = glob.glob(os.path.join(orig_dir, pattern))
    
    # Track statistics
    differences = []
    max_diff_pct = 0
    max_diff_file = ""
    
    print(f"{'File':<20} {'Max Diff':<15} {'Avg Diff':<15} {'Diff %':<15}")
    print("-" * 70)
    
    # Process each file
    for orig_path in sorted(orig_files):
        # Get the member number to find matching file in new directory
        base_name = os.path.basename(orig_path)
        member_num = int(base_name.split('_')[1])
        new_path = os.path.join(new_dir, base_name)
        
        # Check if the new file exists
        if not os.path.exists(new_path):
            print(f"{base_name:<20} MISSING IN NEW DIRECTORY")
            continue
        
        # Load matrices from both files
        orig_matrix = np.loadtxt(orig_path, delimiter=',')
        new_matrix = np.loadtxt(new_path, delimiter=',')
        
        # Calculate differences
        abs_diff = np.abs(new_matrix - orig_matrix)
        max_abs_diff = np.max(abs_diff)
        avg_abs_diff = np.mean(abs_diff)
        
        # Calculate percentage difference based on original values
        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            rel_diff = np.where(orig_matrix != 0, 
                               abs_diff / np.abs(orig_matrix) * 100, 
                               0)
        
        # Replace NaN or inf with 0
        rel_diff = np.nan_to_num(rel_diff)
        max_rel_diff = np.max(rel_diff)
        
        # Track highest difference
        if max_rel_diff > max_diff_pct:
            max_diff_pct = max_rel_diff
            max_diff_file = base_name
        
        differences.append({
            'member': member_num,
            'max_abs_diff': max_abs_diff,
            'avg_abs_diff': avg_abs_diff,
            'max_rel_diff': max_rel_diff
        })
        
        print(f"{base_name:<20} {max_abs_diff:<15.4f} {avg_abs_diff:<15.4f} {max_rel_diff:<15.2f}%")
    
    print("\nSummary:")
    print(f"Largest difference: {max_diff_pct:.2f}% in {max_diff_file}")
    
    # Optional: Plot differences
    if differences:
        df = pd.DataFrame(differences)
        plt.figure(figsize=(10, 6))
        plt.bar(df['member'], df['max_rel_diff'])
        plt.xlabel('Member Number')
        plt.ylabel('Maximum Difference (%)')
        plt.title('Stiffness Matrix Differences Between Original and New')
        plt.savefig('stiffness_comparison.png')
        print("Comparison plot saved as 'stiffness_comparison.png'")

if __name__ == "__main__":
    compare_stiffness_matrices()
