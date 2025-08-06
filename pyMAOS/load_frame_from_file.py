import os
import sys

import pint

from rx.subject.subject import Subject

import pyMAOS

from pymaos_linear_elastic_material import get_materials_from_yaml





# Alternative: show all imported classes
# print_imported_classes()

from pyMAOS.logger import setup_logger

# Import all unit-related functionality from units.py module
# from pyMAOS.units_mod import (
#     # FORCE_UNIT, LENGTH_UNIT, MOMENT_UNIT, PRESSURE_UNIT, DISTRIBUTED_LOAD_UNIT,  # Display units
#     # INTERNAL_FORCE_UNIT, INTERNAL_LENGTH_UNIT, INTERNAL_MOMENT_UNIT, INTERNAL_PRESSURE_UNIT,  # Internal units
#     # INTERNAL_DISTRIBUTED_LOAD_UNIT, INTERNAL_PRESSURE_UNIT_EXPANDED,  # More internal units
#     update_units_from_json,  set_unit_system, INTERNAL_DISTRIBUTED_LOAD_UNIT  # Functions
# )

# from pyMAOS.units_mod import SI_UNITS, IMPERIAL_UNITS, METRIC_KN_UNITS

# Import other modules
from pyMAOS.node2d import R2Node

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


def load_frame_from_file(filename, logger=None, schema_file=None, show_vtk=False):
    """
    Reads a structural model from a JSON or YAML file and creates nodes and elements using internal units

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
        (node_list, element_list) ready for structural analysis, in internal units
    """

    # Import the unit_manager directly from the module
    import pyMAOS

    # Get internal units
    internal_length_unit = pyMAOS.unit_manager.INTERNAL_LENGTH_UNIT
    internal_force_unit = pyMAOS.unit_manager.INTERNAL_FORCE_UNIT
    internal_moment_unit = pyMAOS.unit_manager.INTERNAL_MOMENT_UNIT

    system_name = pyMAOS.unit_manager.system_name

    log(f"Using internal unit system: {system_name}")
    log(f"Internal length unit: {internal_length_unit}")
    log(f"Internal force unit: {internal_force_unit}")
    log(f"Internal moment unit: {internal_moment_unit}")
    # log(f"Using UnitManager registry with id: {id(pyMAOS.unit_manager.ureg)}")

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
                from pyMAOS.json_utils import validate_input_with_schema
                validate_input_with_schema(filename, schema_file=schema_file)
                log("JSON validation passed!")
            except Exception as e:
                log(f"Warning: JSON validation failed: {e}")

        # Load JSON data
        with open(filename, 'r', encoding='utf-8') as file:
            data = json.load(file)

    # Process nodes - convert to internal units (not hardcoded to meters)
    nodes_dict = {}
    for node_data in data.get("nodes", []):
        node_id = node_data["id"]

        # Use unit_manager to parse coordinates with potential units
        x = pyMAOS.unit_manager.parse_value(str(node_data["x"]))
        y = pyMAOS.unit_manager.parse_value(str(node_data["y"]))

        # Convert to internal length units if units are specified
        x_internal = x.to(internal_length_unit) if isinstance(x, pint.Quantity) else x
        y_internal = y.to(internal_length_unit) if isinstance(y, pint.Quantity) else y

        node = R2Node(node_id, x_internal, y_internal)
        nodes_dict[node_id] = node

    log(f"Read {len(nodes_dict)} nodes.")
    log(str(nodes_dict))

    # Process node supports
    log("\nProcessing node supports:")
    for node_restraint in data.get("supports", []):
        node_id = node_restraint["node"]
        if node_id not in nodes_dict:
            log(f"Warning: Restraint specified for non-existent node {node_id}")
            continue

        # Get restraint values (1=fixed, 0=free)
        ux = int(node_restraint.get("ux", 0))
        uy = int(node_restraint.get("uy", 0))
        rz = int(node_restraint.get("rz", 0))

        # Apply restraints to the node
        nodes_dict[node_id].restraints = [ux, uy, rz]
        log(f"Node {node_id} restraints: Ux={ux}, Uy={uy}, Rz={rz}")
    # Process joint loads - convert to internal units
    for joint_load in data.get("joint_loads", []):
        node_id = joint_load["node"]

        # Parse forces with potential units
        fx_with_units = pyMAOS.unit_manager.parse_value(str(joint_load.get("fx", 0)))
        fy_with_units = pyMAOS.unit_manager.parse_value(str(joint_load.get("fy", 0)))
        mz_with_units = pyMAOS.unit_manager.parse_value(str(joint_load.get("mz", 0)))

        # Convert to internal units (not hardcoded SI)
        fx = fx_with_units.to(internal_force_unit) if isinstance(fx_with_units,
                                                                           pint.Quantity) else fx_with_units
        fy = fy_with_units.to(internal_force_unit) if isinstance(fy_with_units,
                                                                           pint.Quantity) else fy_with_units
        mz = mz_with_units.to(internal_moment_unit) if isinstance(mz_with_units,
                                                                            pint.Quantity) else mz_with_units

        if node_id in nodes_dict:
            # Store in internal units
            nodes_dict[node_id].add_nodal_load(fx, fy, mz, "D")
            log(f"Node {node_id} load: Fx={fx:.4g} {internal_force_unit}, Fy={fy:.4g} {internal_force_unit}, Mz={mz:.4g} {internal_moment_unit}")

    # Process materials
    try:
        materials_yml = os.path.join(os.path.dirname(filename), "materials.yml")
        if not os.path.exists(materials_yml):
            materials_yml = os.path.join("materials.yml")
        materials_dict=get_materials_from_yaml(materials_yml)
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
        from pymaos_sections import get_sections_from_yaml
        sections_dict=get_sections_from_yaml(sections_yml, logger=logger)

    except Exception as e:
        log(f"Error loading sections: {e}")
        raise

    # Process members/elements
    element_list = []
    elements_dict = {}
    for member_data in data.get("members", []):
        member_id = member_data["id"]
        i_node = member_data["i_node"]

        inode=nodes_dict[i_node]
        inode.is_inode_of_elem_ids.append(member_id)


        j_node = member_data["j_node"]
        jnode = nodes_dict[j_node]
        jnode.is_jnode_of_elem_ids.append(member_id)


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

        inode.is_inode_of_elems.append(element); jnode.is_inode_of_elems.append(element)

    log(f"Read {len(element_list)} elements.")

    # Process joint loads - always convert to SI units
    for joint_load in data.get("joint_loads", []):
        node_id = joint_load["node"]

        # Parse forces with potential units
        fx_with_units = pyMAOS.unit_manager.parse_value(str(joint_load.get("fx", 0)))
        fy_with_units = pyMAOS.unit_manager.parse_value(str(joint_load.get("fy", 0)))
        mz_with_units = pyMAOS.unit_manager.parse_value(str(joint_load.get("mz", 0)))

        # Convert to SI units (Newtons, Newton-meters)
        fx = fx_with_units.to('N').magnitude if isinstance(fx_with_units, pint.Quantity) else fx_with_units
        fy = fy_with_units.to('N').magnitude if isinstance(fy_with_units, pint.Quantity) else fy_with_units
        mz = mz_with_units.to('N*m').magnitude if isinstance(mz_with_units, pint.Quantity) else mz_with_units

        if node_id in nodes_dict:
            # Store in SI units
            nodes_dict[node_id].add_nodal_load(fx, fy, mz, "D")
            log(f"Node {node_id} load: Fx={fx:.4g}, Fy={fy:.4g}, Mz={mz:.4g}")

    # Import the necessary load classes

    log(f"\nProcessing {len(data.get('member_loads', []))} member loads:")

    # Process member loads - always convert to SI units
    for member_load in data.get("member_loads", []):

        pprint(member_load)
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

            from pyMAOS import INTERNAL_LENGTH_UNIT, INTERNAL_DISTRIBUTED_LOAD_UNIT
            w1_with_units = pyMAOS.unit_manager.parse_value(str(member_load.get("wi", 0))).to(INTERNAL_DISTRIBUTED_LOAD_UNIT)

            if "wj" in member_load:
                w2_with_units = pyMAOS.unit_manager.parse_value(str(member_load["wj"])).to(INTERNAL_DISTRIBUTED_LOAD_UNIT)
            else:
                w2_with_units = w1_with_units

            # Get positions - check for percentage parameters first
            if "a_pct" in member_load:
                a_pct = float(member_load["a_pct"])
                a_with_units = a_pct / 100.0 * element.length
                log(f"  Using a_pct={a_pct}% → position a={a_with_units:.4f}")
            else:
                a_with_units = pyMAOS.unit_manager.parse_value(str(member_load["a"])).to(INTERNAL_LENGTH_UNIT)

            if "b_pct" in member_load:
                b_pct = float(member_load["b_pct"])
                b_with_units = b_pct / 100.0 * element.length
                log(f"  Using b_pct={b_pct}% → position b={b_with_units:.4f}")
            else:
                b_with_units = pyMAOS.unit_manager.parse_value(member_load.get("b", element.length)).to(INTERNAL_LENGTH_UNIT)

            element.add_distributed_load(w1_with_units, w2_with_units, a_with_units, b_with_units, load_case, direction=direction)

        elif load_type == 1:  # Point load
            # Parse force magnitude with unit conversion

            from pyMAOS import INTERNAL_LENGTH_UNIT, INTERNAL_FORCE_UNIT
            p_with_units = pyMAOS.unit_manager.parse_value(str(member_load.get("p", 0))).to(INTERNAL_FORCE_UNIT)
            # Parse position - use percentage value if available
            if "a_pct" in member_load:
                a_pct = float(member_load["a_pct"])
                a_with_units = a_pct / 100.0 * element.length
            else:
                a_with_units = pyMAOS.unit_manager.parse_value(str(member_load["a"])).to(INTERNAL_LENGTH_UNIT)
            if a_with_units > element.length:
                log(f"Warning: Point load position {a_with_units} exceeds element length {element.length}. Clamping to length.")
                a_with_units = element.length
            # Remove b_with_units if it exists
            if 'b_with_units' in locals():
                del b_with_units

            # Apply the load to the element with correct direction
            if direction == "X":
                element.add_point_load(p_with_units, a_with_units, load_case, direction="xx")
            else:
                element.add_point_load(p_with_units, a_with_units, load_case)

        elif load_type == 2:  # Point moment
            # Parse moment magnitude with unit conversion
            internal_length_unit = pyMAOS.unit_manager.INTERNAL_LENGTH_UNIT
            m_with_units = pyMAOS.unit_manager.parse_value(str(member_load.get(internal_length_unit, 0)))

            # Parse position - use percentage value if available
            if "a_pct" in member_load:
                a_pct = float(member_load["a_pct"])
                a = a_pct / 100.0 * element.length
            else:
                a = float(member_load.get("a", 0.0))

            # Convert to SI units (N·m)
            m = m_with_units.to('N*m').magnitude if isinstance(m_with_units, pint.Quantity) else m_with_units

            # Log with SI units
            log(f"  Element {element_id}: Point moment m={m:.4g}, position={a:.4f}")

            # Apply moment with SI units
            element.add_point_moment(m, a, load_case)

        elif load_type == 4:  # Axial load
            internal_force_unit = pyMAOS.unit_manager.INTERNAL_FORCE_UNIT
            # Parse axial load with unit conversion
            p_with_units = pyMAOS.unit_manager.parse_value(str(member_load.get("p", 0)))

            # Convert to SI units (N)
            p = p_with_units.to(internal_force_unit).magnitude if isinstance(p_with_units, pint.Quantity) else p_with_units

            # Log with SI units
            log(f"  Element {element_id}: Axial load p={p:.4g}")

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
    # log("\n\n--- Node Restraints Summary ---")
    # log("Node ID  |  Ux  |  Uy  |  Rz")
    # log("-" * 30)
    # for node in node_list:
    #     rx, ry, rz = node.restraints
    #     rx_status = "Fixed" if rx == 1 else "Free"
    #     ry_status = "Fixed" if ry == 1 else "Free"
    #     rz_status = "Fixed" if rz == 1 else "Free"
    #     log(f"Node {node.uid:2d}  |  {rx_status:5s} |  {ry_status:5s} |  {rz_status:5s}")

    # Plot structure if requested
    # if show_vtk:
    #     plot_structure_vtk(node_list, element_list, scaling=default_scaling)

    return node_list, element_list

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
    w1_with_units = unit_manager.parse_value(str(member_load.get("wi", 0)))
    from pyMAOS.pymaos_units import INTERNAL_DISTRIBUTED_LOAD_UNIT
    w1 = w1_with_units.to(INTERNAL_DISTRIBUTED_LOAD_UNIT).magnitude if isinstance(w1_with_units,
                                                                                  pint.Quantity) else w1_with_units

    if "wj" in member_load:
        w2_with_units = unit_manager.parse_value(str(member_load["wj"]))
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


from pyMAOS.structure2d_to_json import export_results_to_json


# Example usage
if __name__ == "__main__":

    # Show all pyMAOS classes
    # print_imported_classes('pyMAOS')
    # pprint(globals())
    status = 0
    import argparse

    """Command line interface for unit conversion."""
    parser = argparse.ArgumentParser(description="Convert structural model to SI units")
    parser.add_argument("--input", required=True, help="Input file path (.yml, .json, or .bin)")
    parser.add_argument("-w", "--working_dir", default=None, help="Working directory for output files")
    parser.add_argument("--units", choices=["si", "imperial", "metric_kn"], default="imperial",
                        help="Unit system to use (si, imperial, or metric_kn)")
    parser.add_argument("--output", type=str, default="ALL",
                        help="Output format: json, xlsx, html, dash, csv or all (default, case insensitive)")
    parser.add_argument("--vtk", action="store_true",
                        help="Enable VTK visualization of the structure and results")

    args = parser.parse_args()

    # Use the directory of the input file as the working directory for all outputs
    input_file = os.path.abspath(args.input)
    file_ext = os.path.splitext(input_file)[1].lower()
    base_name = os.path.splitext(input_file)[0]

    if args.working_dir:
        working_dir = args.working_dir
        os.makedirs(working_dir, exist_ok=True)  # Create the directory if it doesn't exist
    else:
        working_dir = os.path.dirname(input_file) or os.path.curdir

    print(f"Working directory set to: {working_dir}")
    print(f"Current directory: {os.path.abspath(os.getcwd())}")

    from pyMAOS import info,warning,error,log_exception
    logfile = f"{base_name}.log"
    logger = setup_logger('pyMAOS', logfile)
    logger.info(f"Using working directory: {working_dir}")

    # global DATADIR
    # DATADIR = os.environ.get('DATADIR', working_dir)

    from pyMAOS.pymaos_units import IMPERIAL_DISPLAY_UNITS, SI_UNITS, METRIC_KN_UNITS

    # Choose the unit system with a simple function call
    logger.info(f"\nSetting {args.units} unit system...")
    from pyMAOS import unit_manager
    if args.units == "imperial":
        unit_manager.ureg.default_system = args.units
        unit_manager.ureg.default_preferred_units = IMPERIAL_DISPLAY_UNITS
        unit_manager.set_display_unit_system(IMPERIAL_DISPLAY_UNITS, args.units)
        unit_manager.setup_preferred_units(system_name=args.units)
        logger.info("Using imperial unit system")
    elif args.units == "si":
        unit_manager.set_display_unit_system(SI_UNITS, args.units)
        logger.info("Using SI unit system")
    elif args.units == "metric_kn":
        unit_manager.set_display_unit_system(METRIC_KN_UNITS, args.units)
        logger.info("Using metric kN unit system")

    # Get current unit system directly from the manager
    from pprint import pprint
    current_units = pyMAOS.unit_manager.get_current_units(); pprint(current_units)

    system_name = pyMAOS.unit_manager.get_system_name(); print(system_name)

    # Import the R2Structure class
    from pyMAOS.structure2d import R2Structure  # Instead of from pyMAOS.R2Structure import R2Structure
    from frame2d import R2Frame
    R2Frame.plot_enabled = False

    loadcombo = LoadCombo("D", {"D": 1.0}, ["D"], False, "SLS")
    # Process based on file extension
    if file_ext == '.bin':
        logger.info(f"Loading structure state from binary file: {input_file}")
        model_structure = R2Structure([], [])  # Create empty structure initially

        if model_structure.load_structure_state(input_file):
            logger.info("Successfully loaded structure state")


            try:
                from pyMAOS.plotting.structure2d_matplotlib import plot_structure_matplotlib

                try:
                    fig, ax=plot_structure_matplotlib(model_structure.nodes, model_structure.members)
                    fig.show()
                except Exception as e:
                    logger.error(e)

                from pyMAOS.plotting.structure2d_vista import plot_structure_pv
                scaling_file=os.path.join(os.getcwd(), 'pyMAOS', "plotting", "scaling.json")
                plot_structure_pv(model_structure, loadcombo, scaling_file=scaling_file)


                # Plot the structure with deformations
                scaling = {"displacement": 100}  # Adjust scaling as needed
                # Example usage
                from pyMAOS.plotting.structure2d_matplotlib import plot_deformed_structure
                fig, ax = plot_deformed_structure(structure.nodes, structure.members, all_loads, scaling)

                # Show the plot
                fig.show()

                # Optional VTK plot
                if args.vtk:
                    try:
                        from structure2d_vtk import plot_structure_vtk

                        plot_structure_pv(model_structure, loadcombo)
                    except Exception as e:
                        from pyMAOS import error

                        error(f"Could not plot with VTK: {str(e)}")
            except Exception as e:
                logger.error(f"Error in visualization: {str(e)}")
        else:
            logger.error("Failed to load structure state from binary file")
            sys.exit(1)
    else:  # YAML, JSON or other formats
        logger.info(f"Loading structural model from file: {input_file}")
        structure_state_bin = f"{base_name}.bin"

        try:
            # Load and process the structural model
            node_list, element_list = load_frame_from_file(input_file, logger=logger, show_vtk=args.vtk)
            logger.info(f"Total nodes: {len(node_list)}")
            logger.info(f"Total elements: {len(element_list)}")

            # Create and solve the model
            model_structure = R2Structure(node_list, element_list)
            logger.info("Solving linear static problem...")

            try:
                U = model_structure.solve_linear_static(loadcombo, output_dir=working_dir,
                                                        structure_state_bin=structure_state_bin, verbose=True)
                model_structure.set_node_displacements(loadcombo)
                model_structure.compute_reactions(loadcombo)

                structure_state_bin = f"{base_name}.bin"
                model_structure.save_structure_state(structure_state_bin)
                logger.info(f"Structure state saved to {structure_state_bin}")

            except Exception as e:
                from pyMAOS.logger import log_exception

                log_exception(logger, message=f"Error solving linear static problem: {str(e)}")
                sys.exit(1)

        except Exception as e:
            from pyMAOS.logger import log_exception

            log_exception(logger, message=f"Error loading structural model: {str(e)}")
            sys.exit(1)

    logger.info("Linear static problem solved successfully.")

    # logger.info(f"Displacements U:\n{U}")
    # logger.info(str(model_structure))

    # # Save displacement results
    # np.save(os.path.join(working_dir, 'U.npy'), U)
    # np.savetxt(os.path.join(working_dir, 'U.txt'), U)
    from pyMAOS.plotting.structure2d_vista import plot_structure_py

    # Example call with input directory
    plot_structure_pv(model_structure, loadcombo, scaling_file=None)

    # Export results to JSON if requested
    # Convert to uppercase to make it case-insensitive
    output_format = args.output.upper()

    # Validate the output format after conversion to uppercase
    valid_formats = ["JSON", "XLSX", "HTML", "DASH", "CSV", "ALL"]
    if output_format not in valid_formats:
        print(f"Error: Invalid output format '{args.output}'. Valid options are: {', '.join(valid_formats).lower()}")
        sys.exit(1)

    # Output format handling
    output_to_json = output_format in ["JSON", "ALL"]
    output_to_xlsx = output_format in ["XLSX", "ALL"]
    output_to_html = output_format in ["HTML", "ALL"]
    output_to_dash = output_format in ["DASH", "ALL"]
    output_to_csv = output_format in ["CSV", "ALL"]

    # Processing for JSON output
    if output_to_json:
        json_output = f"{os.path.splitext(input_file)[0]}_results.json"
        export_results_to_json(model_structure, [loadcombo], json_output)
        logger.info(f"\nResults exported in {pyMAOS.unit_manager.system_name} units: {json_output}")

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

        # Export results to Dash if requested
        if output_to_html:
            results_html = f"{os.path.splitext(input_file)[0]}_results.html"
            try:
                # Export results to Excel with proper unit system
                from pyMAOS.structure2d_to_html import generate_html_report
                R2Structure.generate_html_report=generate_html_report
                model_structure.generate_html_report(loadcombos=[loadcombo],
                                                        output_html=results_html)
                logger.info(f"Results exported to Excel: {results_xlsx} using {args.units} units")
            except Exception as e:
                logger.error(f"Error exporting results to Excel: {e}")
                import traceback

                traceback.print_exc()

        # Processing for Dash output
        if output_to_dash:
            results_dash = f"{os.path.splitext(input_file)[0]}_dashboard.html"
            try:
                from pyMAOS.structure2d_to_dash import generate_dash_report

                generate_dash_report(model_structure, [loadcombo], output_file=results_dash)
                logger.info(f"Results exported to Dash dashboard: {results_dash}")
            except Exception as e:
                logger.error(f"Error exporting results to Dash: {e}")
                import traceback

                traceback.print_exc()

        # Processing for CSV output
        if output_to_csv:
            csv_dir = f"{os.path.splitext(input_file)[0]}_csv"
            os.makedirs(csv_dir, exist_ok=True)
            try:
                from pyMAOS.structure2d_to_csv import export_results_to_csv
                R2Structure.export_results_to_csv=export_results_to_csv
                model_structure.export_results_to_csv(csv_dir, loadcombos=[loadcombo])
                logger.info(f"Results exported to CSV files in: {csv_dir}")
            except Exception as e:
                logger.error(f"Error exporting results to CSV: {e}")
                import traceback

                traceback.print_exc()

    # Visualize results only if --vtk flag is used
    if args.vtk:
        logger.info("Showing VTK visualization...")
        model_structure.plot_loadcombos_vtk(loadcombos=None, scaling=default_scaling)

    # Pause the program before exiting
    logger.info("\n\nAnalysis complete. Press Enter to exit...")
