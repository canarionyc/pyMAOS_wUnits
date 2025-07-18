from ast import Constant
import os
from os import path
import sys
import json
import numpy as np
from contextlib import redirect_stdout
import pint

# Print module search paths and loading information
print("\n=== Python Module Search Paths ===")
print(f"Current working directory: {os.getcwd()}")
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")
print("\nModule search paths (sys.path):")
for i, p in enumerate(sys.path):
    print(f"  {i}: {p}")
print("=" * 40 + "\n")

# Import all unit-related functionality from units.py module
from pyMAOS.units import (
    ureg, Q_,  # Unit registry 
    DISPLAY_UNITS, FORCE_UNIT, LENGTH_UNIT, MOMENT_UNIT, PRESSURE_UNIT, DISTRIBUTED_LOAD_UNIT,  # Display units
    INTERNAL_FORCE_UNIT, INTERNAL_LENGTH_UNIT, INTERNAL_MOMENT_UNIT, INTERNAL_PRESSURE_UNIT,  # Internal units
    INTERNAL_DISTRIBUTED_LOAD_UNIT, INTERNAL_PRESSURE_UNIT_EXPANDED,  # More internal units
    update_units_from_json, parse_value_with_units  # Functions
)

# Configure numpy printing
np.set_printoptions(precision=4, suppress=False, floatmode='maxprec_equal')

# Add custom formatters for different numeric types
def format_with_dots(x) -> str: return '.'.center(12) if abs(x) < 1e-10 else f"{x:<12.4g}"
def format_double(x) -> str: return '.'.center(16) if abs(x) < 1e-10 else f"{x:<16.8g}"  # More precision for doubles
np.set_printoptions(formatter={
    np.float64: format_double,
    np.float32: format_with_dots
}) # type: ignore

# Import other modules
from pyMAOS.plot_structure import plot_structure_vtk
from pyMAOS.nodes import R2Node
from pyMAOS.Frame import R2Frame
from pyMAOS.material import LinearElasticMaterial as Material
from pyMAOS.section import Section
import pyMAOS.R2Structure as R2Struct
from pyMAOS.loadcombos import LoadCombo

default_scaling = {
        "axial_load": 100,
        "normal_load": 100,
        "point_load": 1,
        "axial": 2,
        "shear": 2,
        "moment": 0.1,
        "rotation": 5000,
        "displacement": 100,
    }

def load_frame_from_json(filename):
    """
    Reads a structural model from a JSON file
    
    Parameters
    ----------
    filename : str
        Path to the input JSON file
        
    Returns
    -------
    tuple
        (node_list, element_list) ready for structural analysis
    """
    with open(filename, 'r', encoding='utf-8') as file:
        data = json.load(file)
    
    # Process units if present
    if "units" in data:
        update_units_from_json(json.dumps(data["units"]))
    
    # Process nodes
    nodes_dict = {}
    for node_data in data.get("nodes", []):
        node_id = node_data["id"]
        
        # Parse coordinates with potential units
        x = parse_value_with_units(str(node_data["x"]))
        y = parse_value_with_units(str(node_data["y"]))
        
        # Convert to meters if units are specified
        x_meters = x.to('m').magnitude if isinstance(x, pint.Quantity) else x
        y_meters = y.to('m').magnitude if isinstance(y, pint.Quantity) else y
        
        node = R2Node(node_id, x_meters, y_meters)
        nodes_dict[node_id] = node
    
    print(f"Read {len(nodes_dict)} nodes."); print(nodes_dict)
    
    # Process supports
    for support_data in data.get("supports", []):
        node_id = support_data["node"]
        rx = support_data["rx"]
        ry = support_data["ry"]
        rz = support_data["rz"]
        
        if node_id in nodes_dict:
            nodes_dict[node_id].restraints = [rx, ry, rz]
            print(f"Node {node_id} supports: rx={rx}, ry={ry}, rz={rz}")
    
    # Process materials
    def process_materials(materials_data) -> dict:
        materials_dict = {}
        for material_data in materials_data:
            material_id = material_data["id"]
            material_type = material_data.get("type", "linear")

        # Parse E with potential units
        e_value_with_units = parse_value_with_units(str(material_data["E"]))
        
        # Convert to standard pressure units
        if isinstance(e_value_with_units, pint.Quantity):
            try:
                # Try converting to Pascal (SI unit) first
                e_value = e_value_with_units.to('Pa').magnitude
                
                # Show both internal value and display value
                display_value = e_value_with_units.to(PRESSURE_UNIT).magnitude
                print(f"Converted material E from {e_value_with_units} to {e_value} Pa")
                print(f"  (Equivalent to {display_value} {PRESSURE_UNIT} in display units)")
            except Exception as e1:
                try:
                    # Try the expanded form as a fallback
                    e_value = e_value_with_units.to(INTERNAL_PRESSURE_UNIT_EXPANDED).magnitude
                    print(f"Converted material E to {e_value} Pa using expanded form N/m²")
                except Exception as e2:
                    # If all conversions fail, use raw value
                    print(f"Warning: Could not convert {e_value_with_units} to Pa or N/m², using raw value")
                    e_value = e_value_with_units.magnitude
        else:
            e_value = e_value_with_units
            print(f"No units specified for material E, assuming {e_value} Pa")
        
        material = Material(uid=material_id, E=float(e_value))  # Explicitly cast to float
        materials_dict[material_id] = material
    
    materials_dict=process_materials(data.get("materials", []))

    # Process sections
    sections_dict = {}
    for section_data in data.get("sections", []):
        section_id = section_data["id"]
        
        # Parse area and ixx with potential units
        area_with_units = parse_value_with_units(str(section_data["area"]))
        ixx_with_units = parse_value_with_units(str(section_data["ixx"]))
        
        # Convert to standard units
        if isinstance(area_with_units, pint.Quantity):
            try:
                area = float(area_with_units.to('m^2').magnitude)  # Explicitly cast to float
                print(f"Converted section area from {area_with_units} to {area} m²")
            except:
                area = float(area_with_units.magnitude)  # Explicitly cast to float
                print(f"Warning: Could not convert {area_with_units} to m², using raw value")
        else:
            area = float(area_with_units)  # Explicitly cast to float
        
        if isinstance(ixx_with_units, pint.Quantity):
            try:
                ixx = float(ixx_with_units.to('m^4').magnitude)  # Explicitly cast to float
                print(f"Converted section inertia from {ixx_with_units} to {ixx} m⁴")
            except:
                ixx = float(ixx_with_units.magnitude)  # Explicitly cast to float
                print(f"Warning: Could not convert {ixx_with_units} to m⁴, using raw value")
        else:
            ixx = float(ixx_with_units)  # Explicitly cast to float
        
        section = Section(uid=section_id, Area=area, Ixx=ixx)
        sections_dict[section_id] = section
    
    # Process members/elements
    element_list = []
    elements_dict = {}
    for member_data in data.get("members", []):
        member_id = member_data["id"]
        i_node = member_data["i_node"]
        j_node = member_data["j_node"]
        mat_id = member_data["material"]
        sec_id = member_data["section"]
        
        # Create frame element
        element = R2Frame(
            uid=member_id,
            inode=nodes_dict[i_node],
            jnode=nodes_dict[j_node],
            material=materials_dict[mat_id],
            section=sections_dict[sec_id]
        )
        element_list.append(element)
        elements_dict[member_id] = element
    
    print(f"Read {len(element_list)} elements.")
    
    # Process joint loads
    for joint_load in data.get("joint_loads", []):
        node_id = joint_load["node"]
        
        # Parse forces with potential units
        fx_with_units = parse_value_with_units(str(joint_load.get("fx", 0)))
        fy_with_units = parse_value_with_units(str(joint_load.get("fy", 0)))
        mz_with_units = parse_value_with_units(str(joint_load.get("mz", 0)))
        
        # Convert to INTERNAL units (Newtons, Newton-meters) for storage and calculation
        fx = fx_with_units.to(INTERNAL_FORCE_UNIT).magnitude if isinstance(fx_with_units, pint.Quantity) else fx_with_units
        fy = fy_with_units.to(INTERNAL_FORCE_UNIT).magnitude if isinstance(fy_with_units, pint.Quantity) else fy_with_units
        mz = mz_with_units.to(INTERNAL_MOMENT_UNIT).magnitude if isinstance(mz_with_units, pint.Quantity) else mz_with_units
        
        # Calculate display values for reporting only
        fx_display = fx_with_units.to(FORCE_UNIT).magnitude if isinstance(fx_with_units, pint.Quantity) else fx_with_units
        fy_display = fy_with_units.to(FORCE_UNIT).magnitude if isinstance(fy_with_units, pint.Quantity) else fy_with_units
        mz_display = mz_with_units.to(MOMENT_UNIT).magnitude if isinstance(mz_with_units, pint.Quantity) else mz_with_units
        
        if node_id in nodes_dict:
            # Store in SI units
            nodes_dict[node_id].add_nodal_load(fx, fy, mz, "D")
            # Report in display units
            print(f"Node {node_id} load: Fx={fx_display}{FORCE_UNIT} ({fx:.4g} N), "
                  f"Fy={fy_display}{FORCE_UNIT} ({fy:.4g} N), "
                  f"Mz={mz_display}{MOMENT_UNIT} ({mz:.4g} N*m)")   

    # Import the necessary load classes
    from pyMAOS.loading import R2_Point_Load, R2_Linear_Load, R2_Axial_Load, R2_Axial_Linear_Load, R2_Point_Moment
    
    print(f"\nProcessing {len(data.get('member_loads', []))} member loads:")
    
    # Process member loads
    for member_load in data.get("member_loads", []):
        element_id = member_load["member_uid"]
        load_type = member_load["load_type"]
        
        if element_id not in elements_dict:
            print(f"Warning: Member load specified for non-existent element {element_id}")
            continue
        
        element = elements_dict[element_id]
        load_case = member_load.get("case", "D")
        direction = member_load.get("direction", "Y").upper()
        location_percent = member_load.get("location_percent", False)
        
        try:
            if load_type == 3:  # Distributed load
                # Parse load intensity
                w1_with_units = parse_value_with_units(str(member_load.get("wi", 0)))
                
                # Convert to INTERNAL units for storage/calculation
                w1 = w1_with_units.to(INTERNAL_DISTRIBUTED_LOAD_UNIT).magnitude if isinstance(w1_with_units, pint.Quantity) else w1_with_units
                
                # Calculate display values for reporting only
                w1_display = w1_with_units.to(DISTRIBUTED_LOAD_UNIT).magnitude if isinstance(w1_with_units, pint.Quantity) else w1_with_units
                
                # Handle non-uniform loads (wj)
                if "wj" in member_load:
                    w2_with_units = parse_value_with_units(str(member_load["wj"]))
                    w2 = w2_with_units.to(INTERNAL_DISTRIBUTED_LOAD_UNIT).magnitude if isinstance(w2_with_units, pint.Quantity) else w2_with_units
                    w2_display = w2_with_units.to(DISTRIBUTED_LOAD_UNIT).magnitude if isinstance(w2_with_units, pint.Quantity) else w2_with_units
                else:
                    w2 = w1  # Uniform load
                    w2_display = w1_display
                
                # Parse positions
                a = float(member_load.get("a", 0.0))
                b = float(member_load.get("b", element.length))
                
                # Convert percentages to actual positions if needed
                if location_percent:
                    print(f"  Converting positions from percentages: a={a}%, b={b}%")
                    a = a / 100.0 * element.length
                    b = b / 100.0 * element.length

                print(f"  Element {element_id}: Distributed load w1={w1_display}{DISTRIBUTED_LOAD_UNIT} ({w1:.4g} N/m), "
                      f"w2={w2_display}{DISTRIBUTED_LOAD_UNIT} ({w2:.4g} N/m), "
                      f"a={a}{LENGTH_UNIT}, b={b}{LENGTH_UNIT}")
                
                element.add_distributed_load(w1, w2, a, b, load_case, direction=direction)
                
            elif load_type == 1:  # Point load
                # Parse force magnitude
                p_with_units = parse_value_with_units(str(member_load.get("p", 0)))
                
                # Convert to internal units for calculation
                p = p_with_units.to(INTERNAL_FORCE_UNIT).magnitude if isinstance(p_with_units, pint.Quantity) else p_with_units
                
                # Get display value for reporting
                p_display = p_with_units.to(FORCE_UNIT).magnitude if isinstance(p_with_units, pint.Quantity) else p_with_units
                
                # Parse position - use percentage value if available
                if "a_pct" in member_load:
                    a_pct = float(member_load["a_pct"])
                    a = a_pct / 100.0 * element.length
                    print(f"  Converting a_pct={a_pct}% to position a={a:.4f}{LENGTH_UNIT}")
                else:
                    a = float(member_load.get("a", 0.0))
                
                print(f"  Element {element_id}: Point load p={p_display}{FORCE_UNIT} ({p} N), "
                      f"position={a}{LENGTH_UNIT}")
                
                # Apply in appropriate direction with SI units
                if direction.upper() == "X":
                    element.add_point_load(p, a, load_case, direction="xx")
                else:
                    element.add_point_load(p, a, load_case)
        
            elif load_type == 2:  # Point moment
                # Parse moment magnitude
                m_with_units = parse_value_with_units(str(member_load.get("m", 0)))
                
                # Convert to internal units for calculation
                m = m_with_units.to(INTERNAL_MOMENT_UNIT).magnitude if isinstance(m_with_units, pint.Quantity) else m_with_units
                
                # Get display value for reporting
                m_display = m_with_units.to(MOMENT_UNIT).magnitude if isinstance(m_with_units, pint.Quantity) else m_with_units
                
                # Parse position - use percentage value if available
                if "a_pct" in member_load:
                    a_pct = float(member_load["a_pct"])
                    a = a_pct / 100.0 * element.length
                    print(f"  Converting a_pct={a_pct}% to position a={a:.4f}{LENGTH_UNIT}")
                else:
                    a = float(member_load.get("a", 0.0))
                
                print(f"  Element {element_id}: Point moment m={m_display}{MOMENT_UNIT} ({m} N*m), "
                      f"position={a}{LENGTH_UNIT}")
                
                # Apply moment with SI units
                element.add_point_moment(m, a, load_case)
            
            else:
                print(f"  Warning: Unsupported load type {load_type}")
                
        except Exception as e:
            print(f"Error processing member load: {e}")
            print(f"  Details: {type(e).__name__} - {str(e)}")
    
    # Create final node list in sorted order
    node_list = [nodes_dict[uid] for uid in sorted(nodes_dict)]
    
    # Print node restraints
    print("\n\n--- Node Restraints Summary ---")
    print("Node ID  |  Ux  |  Uy  |  Rz")
    print("-" * 30)
    for node in node_list:
        rx, ry, rz = node.restraints
        rx_status = "Fixed" if rx == 1 else "Free"
        ry_status = "Fixed" if ry == 1 else "Free"
        rz_status = "Fixed" if rz == 1 else "Free"
        print(f"Node {node.uid:2d}  |  {rx_status:5s} |  {ry_status:5s} |  {rz_status:5s}")
                
    # Plot structure
    plot_structure_vtk(node_list, element_list, scaling=default_scaling)
    
    return node_list, element_list

def load_frame_from_json_new(filename):
    """
    Reads a structural model from a JSON file, first converting it to SI units
    
    Parameters
    ----------
    filename : str
        Path to the input JSON file
        
    Returns
    -------
    tuple
        (node_list, element_list) ready for structural analysis
    """
    import os
    from pathlib import Path
    import json
    
    try:
        # Import the conversion utility
        sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "examples"))
        from convert_units import convert_json_to_si
        
        # Create output filename with _SI suffix
        input_path = Path(filename)
        si_filename = input_path.with_stem(f"{input_path.stem}_SI").with_suffix('.json')
        
        print(f"Converting {filename} to SI units...")
        
        # Load the original JSON
        with open(filename, 'r') as file:
            data = json.load(file)
        
        # Convert to SI units and save
        convert_json_to_si(data, si_filename)
        print(f"Converted data saved to {si_filename}")
        
        # Load the SI version using the original function
        print(f"Loading SI converted model from {si_filename}...")
        return load_frame_from_json(si_filename)
        
    except ImportError:
        print("Warning: Could not import convert_units.py. Falling back to standard loader.")
        return load_frame_from_json(filename)
    except Exception as e:
        print(f"Error in unit conversion: {str(e)}")
        print("Falling back to standard loader with unit conversions.")
        return load_frame_from_json(filename)

# Example usage
if __name__ == "__main__":
    status=0
    import argparse
    """Command line interface for unit conversion."""
    parser = argparse.ArgumentParser(description="Convert JSON structural model to SI units")
    parser.add_argument("input_file", help="Input JSON file path")
    parser.add_argument("-o", "--output", default=None, help="Output JSON file path")
    args = parser.parse_args()

    working_dir=os.path.curdir
    input_file=args.__contains__('input_file') and args.input_file or "input.json"
    DATADIR=os.environ.get('DATADIR', working_dir)
    import jsonschema
    from pyMAOS.units import validate_input_with_schema
    try:
        schema_file=os.path.join(DATADIR,"myschema.json")
        validate_input_with_schema(input_file, schema_file=schema_file)
        print("Validation passed!")
    except jsonschema.exceptions.ValidationError as e:
        print(f"Validation error: {e}")
    except Exception as e:
        print(f"Error during validation: {e}")    
        
    # Load imperial units from file
    # imperial_units_file = os.path.join(DATADIR, 'imperial_units.json')
    # print(f"\nLoading unit settings from: {imperial_units_file}")
    # try:
    #     with open(imperial_units_file, 'r') as unit_file:
    #         imperial_units = json.load(unit_file)
    #         # Update the units using the imported function
    #         update_units_from_json(json.dumps(imperial_units))
            
    #         # Re-import the global unit variables to get the updated values
    #         from pyMAOS.units import (
    #             DISPLAY_UNITS, FORCE_UNIT, LENGTH_UNIT, MOMENT_UNIT, 
    #             PRESSURE_UNIT, DISTRIBUTED_LOAD_UNIT
    #         )
            
    #         print(f"Successfully loaded imperial units: {imperial_units}")
    #         print(f"Updated display units: Force={FORCE_UNIT}, Length={LENGTH_UNIT}, Pressure={PRESSURE_UNIT}, Moment={MOMENT_UNIT}")
    # except FileNotFoundError:
    #     print(f"Warning: Imperial units file '{imperial_units_file}' not found. Using default units.")
    # except json.JSONDecodeError:
    #     print(f"Warning: Invalid JSON in '{imperial_units_file}'. Using default units.")
    # except Exception as e:
    #     print(f"Warning: Error loading imperial units: {str(e)}. Using default units.")
    
    file_path = os.path.join(working_dir, input_file)
    
    # Choose appropriate loader based on file extension
    if input_file.lower().endswith('.json'):
        print(f"Loading structural model from JSON file: {file_path}")
        node_list, element_list = load_frame_from_json_new(file_path)
    else:
        print(f"Loading structural model from text file: {file_path}")
        exit(1)  # Placeholder for text file loading logic 
    
    print(f"Total nodes: {len(node_list)}")
    print(f"Total elements: {len(element_list)}")

    # Pass all display units to the structure
    model_structure = R2Struct.R2Structure(node_list, element_list, units=DISPLAY_UNITS)
    # print(model_structure)

    # Fix the LoadCombo initialization with proper parameters
    loadcombo = LoadCombo("D", {"D": 1.0}, ["D"], False, "SLS")
    print("Solving linear static problem...")
    U = model_structure.solve_linear_static(loadcombo, output_dir=working_dir, verbose=True)
    print(f"Displacements U:\n{U}")
    print(model_structure)

    # Save displacement results
    np.save(os.path.join(working_dir, 'U.npy'), U)
    np.savetxt(os.path.join(working_dir, 'U.txt'), U)

    model_structure.plot_loadcombos_vtk(loadcombos=None, scaling=default_scaling)
     
    # Pause the program before exiting
    print("\n\nAnalysis complete. Press Enter to exit...")