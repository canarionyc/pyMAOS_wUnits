from ast import Constant
import os
from os import path
import sys

import numpy as np
from contextlib import redirect_stdout
import pint
from rx.subject.subject import Subject
from rx import operators as ops
import rx
import logging  # Add logging module
from pprint import pp, pprint


def is_class_imported(class_name):
    """
    Check if a class has been imported in the current namespace
    
    Parameters
    ----------
    class_name : str
        Name of the class to check
        
    Returns
    -------
    bool
        True if class exists, False otherwise
    """
    return class_name in globals() or class_name in locals()


def check_class_exists(class_name):
    """Check if a class is available in the current namespace"""
    try:
        # Try to evaluate the class name
        return eval(class_name) is not None
    except (NameError, AttributeError):
        return False


def is_class_available_from_module(module_name, class_name):
    """Check if a class is available from a specific imported module"""
    import sys

    # Check if module is imported
    if module_name not in sys.modules:
        return False

    # Get the module object
    module = sys.modules[module_name]

    # Check if class exists in module
    return hasattr(module, class_name)


def list_imported_classes(module_filter=None):
    """
    List all classes that have been imported in the current Python script
    
    Parameters
    ----------
    module_filter : str or list of str, optional
        Filter classes by module name prefix (e.g., 'pyMAOS' or ['pyMAOS', 'numpy'])
        
    Returns
    -------
    dict
        Dictionary mapping class names to their module names
    """
    import inspect
    import sys

    # Normalize filter to a list
    if module_filter is None:
        filters = None
    elif isinstance(module_filter, str):
        filters = [module_filter]
    else:
        filters = list(module_filter)

    classes_dict = {}

    # Check global namespace
    for name, obj in globals().items():
        if inspect.isclass(obj):
            module = inspect.getmodule(obj)
            if module:
                module_name = module.__name__
                if filters is None or any(module_name.startswith(m) for m in filters):
                    classes_dict[name] = module_name

    # Check modules in sys.modules
    for module_name, module in sys.modules.items():
        # Skip None modules or if filtering is active and module doesn't match
        if module is None:
            continue
        if filters and not any(module_name.startswith(m) for m in filters):
            continue

        try:
            for name, obj in inspect.getmembers(module, inspect.isclass):
                # Only include if defined in this module (not imported into it)
                if hasattr(obj, '__module__') and obj.__module__ == module_name:
                    classes_dict[name] = module_name
        except:
            # Some modules might raise errors when inspected
            pass

    return classes_dict


# Example usage:
def print_imported_classes(module_filter=None):
    """Print imported classes in a formatted table"""
    classes = list_imported_classes(module_filter)

    print(f"\n{'Class Name':<30} | {'Module'}")
    print("-" * 60)

    for name, module in sorted(classes.items()):
        print(f"{name:<30} | {module}")

    print(f"\nTotal: {len(classes)} classes found")


# Alternative: show all imported classes
# print_imported_classes()

# Configure logging to write to both console and file
def setup_logger(log_file=None):
    """
    Set up logger to output to both console and file
    
    Parameters
    ----------
    log_file : str, optional
        Path to log file. If None, logging will only be to console.
    """
    # Create logger
    logger = logging.getLogger('pyMAOS')
    logger.setLevel(logging.INFO)

    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    # Create console handler and set level to info
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # Create file handler if log_file is provided
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


# Print module search paths and loading information
print("\n=== Python Module Search Paths ===")
print(f"Current working directory: {os.getcwd()}")
print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")
print("\nModule search paths (sys.path):")
for i, p in enumerate(sys.path):
    print(f"  {i}: {p}")
print("=" * 40 + "\n")

# from pyMAOS.globals import SI_UNITS, IMPERIAL_UNITS, METRIC_KN_UNITS

# Import all unit-related functionality from units.py module
from pyMAOS.units_mod import (
    ureg, Q_,  # Unit registry
    # FORCE_UNIT, LENGTH_UNIT, MOMENT_UNIT, PRESSURE_UNIT, DISTRIBUTED_LOAD_UNIT,  # Display units
    # INTERNAL_FORCE_UNIT, INTERNAL_LENGTH_UNIT, INTERNAL_MOMENT_UNIT, INTERNAL_PRESSURE_UNIT,  # Internal units
    # INTERNAL_DISTRIBUTED_LOAD_UNIT, INTERNAL_PRESSURE_UNIT_EXPANDED,  # More internal units
    update_units_from_json, parse_value_with_units, set_unit_system, INTERNAL_DISTRIBUTED_LOAD_UNIT  # Functions
)

# from pyMAOS.units_mod import SI_UNITS, IMPERIAL_UNITS, METRIC_KN_UNITS

# Configure numpy printing
np.set_printoptions(precision=4, suppress=False, floatmode='maxprec_equal')


# Add custom formatters for different numeric types
def format_with_dots(x) -> str: return '.'.center(12) if abs(x) < 1e-10 else f"{x:<12.4g}"


def format_double(x) -> str: return '.'.center(16) if abs(x) < 1e-10 else f"{x:<16.8g}"  # More precision for doubles


# np.set_printoptions(formatter={
#     np.float64: format_double,
#     np.float32: format_with_dots
# }) # type: ignore
np.set_printoptions(precision=2, threshold=999, linewidth=999, suppress=True,
                    formatter={'all': lambda x: '.'.center(10) if abs(x) < 1e-10 else f"{x:10.4g}"})
# Import other modules
from pyMAOS.plot_structure import plot_structure_vtk
from pyMAOS.node2d import R2Node
from pyMAOS.frame2d import R2Frame
from pyMAOS.material import LinearElasticMaterial as Material
from pyMAOS.section import Section
# import pyMAOS.R2Structure as R2Struct
from pyMAOS.loadcombos import LoadCombo

from pyMAOS.load_utils import LoadConverter

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


def load_frame_from_file(filename, logger=None, schema_file=None, show_vtk=False):
    """
    Reads a structural model from a JSON or YAML file and creates nodes and elements in SI units
    
    Parameters
    ----------
    filename : str
        Path to the input file (JSON or YAML format)
    logger : logging.Logger, optional
        Logger object for output
    schema_file : str, optional
        Path to JSON schema for validation (only used for JSON files)
    show_vtk : bool, optional
        Whether to show VTK plot of the structure
    
    Returns
    -------
    tuple
        (node_list, element_list) ready for structural analysis, all in SI units
    """

    # Use print or logger.info based on what's available
    def log(message):
        if logger:
            logger.info(message)
        else:
            print(message)

    # Check file extension to determine format
    file_ext = os.path.splitext(filename)[1].lower()

    # Create reactive subjects for key data
    model_data = Subject()
    nodes_subject = Subject()
    elements_subject = Subject()
    results_subject = Subject()

    # Load data from file based on format
    if file_ext in ['.yml', '.yaml']:
        try:
            import yaml
            log(f"Loading YAML file: {filename}")
            with open(filename, 'r') as file:
                data = yaml.safe_load(file)
            log("YAML file loaded successfully")
        except ImportError:
            log("Error: PyYAML package not found. Install it using: pip install pyyaml")
            raise
        except Exception as e:
            log(f"Error loading YAML file: {e}")
            raise
    else:  # Default to JSON
        log(f"Loading JSON file: {filename}")
        import json
        # Validate JSON file if schema_file is provided
        if schema_file:
            try:
                from pyMAOS.units_mod import validate_input_with_schema
                validate_input_with_schema(filename, schema_file=schema_file)
                log("JSON validation passed!")
            except Exception as e:
                log(f"Warning: JSON validation failed: {e}")

        # Load JSON data
        with open(filename, 'r', encoding='utf-8') as file:
            data = json.load(file)

    # Process nodes - always convert to SI units (meters)
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

    log(f"Read {len(nodes_dict)} nodes.");
    log(str(nodes_dict))

    # Process supports
    for support_data in data.get("supports", []):
        node_id = support_data["node"]
        rx = support_data["rx"]
        ry = support_data["ry"]
        rz = support_data["rz"]

        if node_id in nodes_dict:
            nodes_dict[node_id].restraints = [rx, ry, rz]
            log(f"Node {node_id} supports: rx={rx}, ry={ry}, rz={rz}")

    # Process materials
    materials_dict = {}
    try:
        materials_yml = os.path.join(os.path.dirname(filename), "materials.yml")
        if not os.path.exists(materials_yml):
            materials_yml = os.path.join("materials.yml")

        log(f"Loading materials from: {materials_yml}")
        with open(materials_yml, 'r') as file:
            # Use unsafe_load to allow object instantiation
            materials_list = yaml.unsafe_load(file)

        # Convert list to dictionary using uid as key
        for material in materials_list:
            materials_dict[material.uid] = material
        log(f"Loaded {len(materials_dict)} materials")
    except Exception as e:
        log(f"Error loading materials: {e}")
        raise

    # Process sections
    sections_dict = {}
    try:
        sections_yml = os.path.join(os.path.dirname(filename), "sections.yml")
        if not os.path.exists(sections_yml):
            sections_yml = os.path.join("sections.yml")

        log(f"Loading sections from: {sections_yml}")
        with open(sections_yml, 'r') as file:
            # Use unsafe_load to allow object instantiation
            sections_list = yaml.unsafe_load(file)

        # Convert list to dictionary using uid as key
        for section in sections_list:
            sections_dict[section.uid] = section
        log(f"Loaded {len(sections_dict)} sections")
    except Exception as e:
        log(f"Error loading sections: {e}")
        raise

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

    log(f"Read {len(element_list)} elements.")

    # Process joint loads - always convert to SI units
    for joint_load in data.get("joint_loads", []):
        node_id = joint_load["node"]

        # Parse forces with potential units
        fx_with_units = parse_value_with_units(str(joint_load.get("fx", 0)))
        fy_with_units = parse_value_with_units(str(joint_load.get("fy", 0)))
        mz_with_units = parse_value_with_units(str(joint_load.get("mz", 0)))

        # Convert to SI units (Newtons, Newton-meters)
        fx = fx_with_units.to('N').magnitude if isinstance(fx_with_units, pint.Quantity) else fx_with_units
        fy = fy_with_units.to('N').magnitude if isinstance(fy_with_units, pint.Quantity) else fy_with_units
        mz = mz_with_units.to('N*m').magnitude if isinstance(mz_with_units, pint.Quantity) else mz_with_units

        if node_id in nodes_dict:
            # Store in SI units
            nodes_dict[node_id].add_nodal_load(fx, fy, mz, "D")
            log(f"Node {node_id} load: Fx={fx:.4g} N, Fy={fy:.4g} N, Mz={mz:.4g} N*m")

    # Import the necessary load classes
    from pyMAOS.loading import R2_Point_Load, R2_Linear_Load, R2_Axial_Load, R2_Axial_Linear_Load, R2_Point_Moment

    log(f"\nProcessing {len(data.get('member_loads', []))} member loads:")

    # Process member loads - always convert to SI units
    for member_load in data.get("member_loads", []):
        element_id = member_load["member_uid"]
        load_type = member_load["load_type"]

        if element_id not in elements_dict:
            log(f"Warning: Member load specified for non-existent element {element_id}")
            continue

        element = elements_dict[element_id]
        load_case = member_load.get("case", "D")
        direction = member_load.get("direction", "Y").upper()

        if load_type == 3:  # Distributed load
            # Extract load intensity parameters with unit conversion
            w1_with_units = parse_value_with_units(str(member_load.get("wi", 0)))

            if "wj" in member_load:
                w2_with_units = parse_value_with_units(str(member_load["wj"]))
            else:
                w2_with_units = w1_with_units

            # Get positions - check for percentage parameters first
            if "a_pct" in member_load:
                a_pct = float(member_load["a_pct"])
                a = a_pct / 100.0 * element.length
                log(f"  Using a_pct={a_pct}% → position a={a:.4f}m")
            else:
                a_with_units = parse_value_with_units(str(member_load["a"]))

            if "b_pct" in member_load:
                b_pct = float(member_load["b_pct"])
                b = b_pct / 100.0 * element.length
                log(f"  Using b_pct={b_pct}% → position b={b:.4f}m")
            else:
                b_with_units = parse_value_with_units(member_load.get("b", element.length))
            from pyMAOS.units_mod import INTERNAL_DISTRIBUTED_LOAD_UNIT
            # Convert to SI units (N/m)
            w1 = w1_with_units.to(INTERNAL_DISTRIBUTED_LOAD_UNIT).magnitude if isinstance(w1_with_units,
                                                                                          pint.Quantity) else w1_with_units
            w2 = w2_with_units.to(INTERNAL_DISTRIBUTED_LOAD_UNIT).magnitude if isinstance(w2_with_units,
                                                                                          pint.Quantity) else w2_with_units

            # Log with SI units
            # log(f"  Element {element_id}: Distributed load w1={w1:.4g} N/m, w2={w2:.4g} N/m, "
            #     f"a={a:.4f} m, b={b:.4f} m")
            from pyMAOS.units_mod import INTERNAL_LENGTH_UNIT
            a = a_with_units.to(INTERNAL_LENGTH_UNIT).magnitude if isinstance(a_with_units,pint.Quantity) else a_with_units
            b = b_with_units.to(INTERNAL_LENGTH_UNIT).magnitude if isinstance(b_with_units,pint.Quantity) else b_with_units
            # Apply the load to the element
            element.add_distributed_load(w1, w2, a, b, load_case, direction=direction)

        elif load_type == 1:  # Point load
            # Parse force magnitude with unit conversion
            from pprint import pprint
            pprint(member_load)
            p_with_units = parse_value_with_units(str(member_load.get("p", 0)))
            from pyMAOS.units_mod import INTERNAL_LENGTH_UNIT, INTERNAL_FORCE_UNIT
            # Parse position - use percentage value if available
            if "a_pct" in member_load:
                a_pct = float(member_load["a_pct"])
                a = a_pct / 100.0 * element.length
            else:
                a_with_units = parse_value_with_units(str(member_load["a"]))
                a = a_with_units.to(INTERNAL_LENGTH_UNIT).magnitude if isinstance(a_with_units,pint.Quantity) else a_with_units

            # Convert to SI units (N)
            p = p_with_units.to(INTERNAL_FORCE_UNIT).magnitude if isinstance(p_with_units, pint.Quantity) else p_with_units

            # Log with SI units
            # log(f"  Element {element_id}: Point load p={p:.4g} N, position={a:.4f} m, direction={direction}")

            # Apply the load to the element with correct direction
            if direction == "X":
                element.add_point_load(p, a, load_case, direction="xx")
            else:
                element.add_point_load(p, a, load_case)

        elif load_type == 2:  # Point moment
            # Parse moment magnitude with unit conversion
            m_with_units = parse_value_with_units(str(member_load.get("m", 0)))

            # Parse position - use percentage value if available
            if "a_pct" in member_load:
                a_pct = float(member_load["a_pct"])
                a = a_pct / 100.0 * element.length
            else:
                a = float(member_load.get("a", 0.0))

            # Convert to SI units (N·m)
            m = m_with_units.to('N*m').magnitude if isinstance(m_with_units, pint.Quantity) else m_with_units

            # Log with SI units
            log(f"  Element {element_id}: Point moment m={m:.4g} N·m, position={a:.4f} m")

            # Apply moment with SI units
            element.add_point_moment(m, a, load_case)

        elif load_type == 4:  # Axial load
            # Parse axial load with unit conversion
            p_with_units = parse_value_with_units(str(member_load.get("p", 0)))

            # Convert to SI units (N)
            p = p_with_units.to('N').magnitude if isinstance(p_with_units, pint.Quantity) else p_with_units

            # Log with SI units
            log(f"  Element {element_id}: Axial load p={p:.4g} N")

            # Apply the axial load
            element.add_axial_load(p, load_case)

        elif load_type == 5:  # Temperature load
            delta_t = float(member_load.get("delta_t", 0))
            alpha = float(member_load.get("alpha", 1.2e-5))

            log(f"  Element {element_id}: Temperature load ΔT={delta_t}°C, α={alpha}/°C")

            # Apply the temperature load if the element supports it
            if hasattr(element, 'add_temperature_load'):
                element.add_temperature_load(delta_t, alpha, load_case)
            else:
                log(f"  Warning: Element {element_id} doesn't support temperature loads")

        else:
            log(f"  Warning: Unsupported load type {load_type}")

        # except Exception as e:
        #     log(f"Error processing member load: {e}")
        #     log(f"  Details: {type(e).__name__} - {str(e)}")

    # Create final node list in sorted order
    node_list = [nodes_dict[uid] for uid in sorted(nodes_dict)]

    # Print node restraints
    log("\n\n--- Node Restraints Summary ---")
    log("Node ID  |  Ux  |  Uy  |  Rz")
    log("-" * 30)
    for node in node_list:
        rx, ry, rz = node.restraints
        rx_status = "Fixed" if rx == 1 else "Free"
        ry_status = "Fixed" if ry == 1 else "Free"
        rz_status = "Fixed" if rz == 1 else "Free"
        log(f"Node {node.uid:2d}  |  {rx_status:5s} |  {ry_status:5s} |  {rz_status:5s}")

    # Plot structure if requested
    if show_vtk:
        plot_structure_vtk(node_list, element_list, scaling=default_scaling)

    return node_list, element_list


def load_frame_from_file_new(filename, logger=None):
    """
    Reads a structural model from a JSON file, first converting it to SI units
    
    Parameters
    ----------
    filename : str
        Path to the input JSON file
    logger : logging.Logger, optional
        Logger object for output
        
    Returns
    -------
    tuple
        (node_list, element_list) ready for structural analysis, all in SI units
    """

    # Use print or logger.info based on what's available
    def log(message):
        if logger:
            logger.info(message)
        else:
            print(message)

    import os
    from pathlib import Path
    import json
    import sys
    try:
        # Import the conversion utility
        sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "examples"))
        from pyMAOS.convert_units import convert_json_to_si

        # Create output filename with _SI suffix
        input_path = Path(filename)
        si_filename = input_path.with_stem(f"{input_path.stem}_SI").with_suffix('.json')

        log(f"Converting {filename} to SI units...")

        # Load the original JSON
        with open(filename, 'r') as file:
            data = json.load(file)

        # Convert to SI units and save
        convert_json_to_si(data, si_filename)
        log(f"Converted data saved to {si_filename}")

        # Load the SI version using the original function
        log(f"Loading SI converted model from {si_filename}...")
        return load_frame_from_file_new(si_filename, logger=logger)

    except ImportError:
        log("Warning: Could not import convert_units.py. Falling back to standard loader.")
        return load_frame_from_file(filename, logger=logger)
    except Exception as e:
        log(f"Error in unit conversion: {str(e)}")
        log("Falling back to standard loader with unit conversions.")
        return load_frame_from_file(filename, logger=logger)


def load_linear_load_reactively(element, member_load, logger=None):
    """Process linear load using reactive approach"""

    # Use print or logger.info based on what's available
    def log(message):
        if logger:
            logger.info(message)
        else:
            print(message)

    from pyMAOS.linear_load_reactive import LinearLoadReactive

    # Create reactive load processor
    load_processor = LinearLoadReactive()

    # Extract load parameters
    w1_with_units = parse_value_with_units(str(member_load.get("wi", 0)))
    from pyMAOS.units_mod import INTERNAL_DISTRIBUTED_LOAD_UNIT
    w1 = w1_with_units.to(INTERNAL_DISTRIBUTED_LOAD_UNIT).magnitude if isinstance(w1_with_units,
                                                                                  pint.Quantity) else w1_with_units

    if "wj" in member_load:
        w2_with_units = parse_value_with_units(str(member_load["wj"]))
        w2 = w2_with_units.to(INTERNAL_DISTRIBUTED_LOAD_UNIT).magnitude if isinstance(w2_with_units,
                                                                                      pint.Quantity) else w2_with_units
    else:
        w2 = w1

    # Get positions - check for percentage parameters first
    if "a_pct" in member_load:
        a_pct = float(member_load["a_pct"])
        a = a_pct / 100.0 * element.length
    else:
        a = float(member_load.get("a", 0.0))

    if "b_pct" in member_load:
        b_pct = float(member_load["b_pct"])
        b = b_pct / 100.0 * element.length
    else:
        b = float(member_load.get("b", element.length))

    # Set parameters in reactive system
    load_processor.set_parameters(w1, w2, a, b, element.length)

    # Subscribe to reactions and use them
    load_processor.reactions.subscribe(
        on_next=lambda reactions: log(f"Calculated reactions: R_i={reactions[0]:.4f}, R_j={reactions[1]:.4f}")
    )

    # Subscribe to constants and use them for the element
    load_processor.constants.pipe(ops.first()).subscribe(
        on_next=lambda constants: element.add_distributed_load(
            w1, w2, a, b, member_load.get("case", "D"),
            direction=member_load.get("direction", "Y").upper()
        )
    )


from pyMAOS.export_utils import export_results_to_json
from pyMAOS.units_mod import unit_manager

# Example usage
if __name__ == "__main__":

    # Show all pyMAOS classes
    print_imported_classes('pyMAOS')
    pp(globals())
    status = 0
    import argparse

    """Command line interface for unit conversion."""
    parser = argparse.ArgumentParser(description="Convert JSON structural model to SI units")
    parser.add_argument("input_file", help="Input JSON file path")
    parser.add_argument("-w", "--working_dir", default=None, help="Working directory for output files")
    parser.add_argument("--units", choices=["si", "imperial", "metric_kn"], default="imperial",
                        help="Unit system to use (si, imperial, or metric_kn)")

    parser.add_argument("--to", choices=["JSON", "XLSX", "BOTH"], default="BOTH",
                        help="Output format: JSON, XLSX, or BOTH (default)")
    parser.add_argument("--vtk", action="store_true",
                        help="Enable VTK visualization of the structure and results")
    args = parser.parse_args()

    # Use the directory of the input file as the working directory for all outputs
    input_file = os.path.abspath(args.input_file)
    if args.working_dir:
        working_dir = args.working_dir
        os.makedirs(working_dir, exist_ok=True)  # Create the directory if it doesn't exist
    else:
        working_dir = os.path.dirname(input_file) or os.path.curdir
    # os.chdir(working_dir)  # Change to the working directory
    print(f"Working directory set to: {working_dir}")
    print(f"Current directory: {os.path.abspath(os.getcwd())}")
    logfile = f"{os.path.splitext(input_file)[0]}.log"
    # Set up logging

    logger = setup_logger(logfile)

    logger.info(f"Using working directory: {working_dir}")
    global DATADIR
    DATADIR = os.environ.get('DATADIR', working_dir)

    from pyMAOS.units_mod import set_unit_system, IMPERIAL_UNITS, SI_UNITS, METRIC_KN_UNITS

    # Choose the unit system with a simple function call
    logger.info(f"\nSetting {args.units} unit system...")
    if args.units == "imperial":
        unit_manager.set_unit_system(IMPERIAL_UNITS, args.units)
        # logger.info("Using imperial unit system")
    elif args.units == "si":
        set_unit_system(SI_UNITS, args.units)
        logger.info("Using SI unit system")
    elif args.units == "metric_kn":
        set_unit_system(METRIC_KN_UNITS, args.units)
        logger.info("Using metric kN unit system")
    # Get current unit system directly from the manager
    current_units = unit_manager.get_current_units();
    pp(current_units)
    system_name = unit_manager.get_system_name();
    print(system_name)
    try:
        logger.info(f"Loading structural model from file: {input_file}")
        # Pass the VTK flag to control visualization in load_frame_from_file
        node_list, element_list = load_frame_from_file(input_file, logger=logger, show_vtk=args.vtk)
    except Exception as e:
        logger.error(f"Error loading structural model: {e}")
        sys.exit(1)

    logger.info(f"Total nodes: {len(node_list)}")
    logger.info(f"Total elements: {len(element_list)}")
    # Check if the R2Structure class is available

    if is_class_imported('R2Structure'):
        print("R2Structure class is available")

    # Check for a class from a specific module
    if is_class_available_from_module('pyMAOS.R2Structure', 'R2Structure'):
        print("R2Structure class is available from pyMAOS.R2Structure module")
    else:
        logger.error("R2Structure class is not available. Please check your installation.")
        sys.exit(1)

    # Import the R2Structure class
    from pyMAOS import R2Structure  # Instead of from pyMAOS.R2Structure import R2Structure

    # Pass all display units to the structure
    model_structure = R2Structure(node_list, element_list)
    # logger.info(model_structure)

    # Fix the LoadCombo initialization with proper parameters
    loadcombo = LoadCombo("D", {"D": 1.0}, ["D"], False, "SLS")
    logger.info("Solving linear static problem...")
    # Solve the linear static problem
    try:
        U = model_structure.solve_linear_static(loadcombo, output_dir=working_dir, verbose=True)
    except Exception as e:
        logger.error(f"Error solving linear static problem: {e}")
        sys.exit(1)
    logger.info("Linear static problem solved successfully.")
    # logger.info(f"Displacements U:\n{U}")
    # logger.info(str(model_structure))

    # # Save displacement results
    # np.save(os.path.join(working_dir, 'U.npy'), U)
    # np.savetxt(os.path.join(working_dir, 'U.txt'), U)

    # Output format handling
    output_to_json = args.to in ["JSON", "BOTH"]
    output_to_xlsx = args.to in ["XLSX", "BOTH"]

    # Export results to JSON if requested
    if output_to_json:
        json_output = f"{os.path.splitext(input_file)[0]}_results.json"
        export_results_to_json(model_structure, [loadcombo], json_output)
        logger.info(f"\nResults exported in SI units: {json_output}")

        # Create a version with display units
        try:
            # Import the conversion utility
            from pyMAOS.convert_units import convert_si_to_display_units

            # Create path for display units version
            json_output_display = f"{os.path.splitext(input_file)[0]}_results_display.json"

            # Get current unit system directly from the manager
            current_units = unit_manager.get_current_units()
            system_name = unit_manager.get_system_name()

            # Convert the SI results to the selected display units
            convert_si_to_display_units(json_output, json_output_display, current_units)
            logger.info(f"Results exported in {system_name} units: {json_output_display}")

        except Exception as e:
            logger.error(f"Error creating display units version: {e}")
            import traceback

            traceback.print_exc()

    # Export results to Excel if requested
    if output_to_xlsx:
        results_xlsx = f"{os.path.splitext(input_file)[0]}_results.xlsx"
        try:
            # Export results to Excel with proper unit system
            model_structure.export_results_to_excel(results_xlsx, loadcombos=[loadcombo],
                                                    unit_system=args.units)
            logger.info(f"Results exported to Excel: {results_xlsx} using {args.units} units")
        except Exception as e:
            logger.error(f"Error exporting results to Excel: {e}")
            import traceback

            traceback.print_exc()

    # Visualize results only if --vtk flag is used
    if args.vtk:
        logger.info("Showing VTK visualization...")
        model_structure.plot_loadcombos_vtk(loadcombos=None, scaling=default_scaling)

    # Pause the program before exiting
    logger.info("\n\nAnalysis complete. Press Enter to exit...")
