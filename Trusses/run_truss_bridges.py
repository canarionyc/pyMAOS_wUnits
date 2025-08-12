import os
import sys
import glob
import matplotlib.pyplot as plt
import shutil
# import argparse
# from pyMAOS import structure2d, frame2d

from pyMAOS.load_frame_from_file import load_frame_from_file
from pyMAOS.plotting.structure2d_matplotlib import plot_structure_matplotlib

def main():
    # Get PYMAOS_HOME environment variable
    pymaos_home = os.environ.get('PYMAOS_HOME')
    if not os.path.isdir(pymaos_home):
        print("Error: PYMAOS_HOME environment variable is not set")
        return 1

    # Define directories
    input_dir = os.path.join(pymaos_home, 'Trusses')
    data_dir = os.path.join(pymaos_home, 'data')

    # Check if directories exist
    if not os.path.isdir(input_dir):
        print(f"Error: Input directory {input_dir} does not exist")
        return 1
    if not os.path.isdir(data_dir):
        print(f"Error: Data directory {data_dir} does not exist")
        return 1

    # Verify materials.yml and sections.yml exist
    materials_file = os.path.join(data_dir, 'materials.yml')
    sections_file = os.path.join(data_dir, 'sections.yml')

    if not os.path.isfile(materials_file) or not os.path.isfile(sections_file):
        print(f"Error: Missing required materials.yml or sections.yml in {data_dir}")
        return 1

    # Copy materials and sections files to input directory
    try:
        print(f"Copying materials and sections files to {input_dir}...")
        shutil.copy2(materials_file, input_dir)
        shutil.copy2(sections_file, input_dir)
    except Exception as e:
        print(f"Error copying files: {e}")
        return 1

    # Find all YAML files in the input directory
    yaml_files = glob.glob(os.path.join(input_dir, '*.yaml'))
    yml_files = glob.glob(os.path.join(input_dir, '*.yml'))
    yaml_files.extend(yml_files)

    # Filter out materials.yml and sections.yml
    yaml_files = [f for f in yaml_files
                 if os.path.basename(f) not in ['materials.yml', 'sections.yml']]

    if not yaml_files:
        print(f"No truss YAML files found in {input_dir}")
        return 0

    print(f"Found {len(yaml_files)} truss YAML files to process")

    # Process each YAML file
    for yaml_file in yaml_files:
        try:
            filename = os.path.basename(yaml_file)
            print(f"Processing {yaml_file}...")

            # Load the structure
            node_list, element_list = load_frame_from_file(yaml_file)
            print(f"  Loaded structure with {len(node_list)} nodes and {len(element_list)} elements")

            # Create plot
            fig, ax = plot_structure_matplotlib(
                node_list,
                element_list,
                show_labels=True,
                node_color='red',
                member_color='blue',
                node_size=60
            )

            # Save as PNG in the same directory
            base_name = os.path.splitext(filename)[0]
            output_file = os.path.join(input_dir, f"{base_name}_plot.png")
            fig.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close(fig)

            print(f"  Saved plot to {output_file}")

        except Exception as e:
            print(f"Error processing {yaml_file}: {e}")
            import traceback
            traceback.print_exc()

    print("\nProcessing complete")
    return 0

if __name__ == "__main__":
    sys.exit(main())