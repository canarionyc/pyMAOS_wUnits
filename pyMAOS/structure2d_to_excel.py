def export_results_to_excel(self, output_file, loadcombos=None, **kwargs):
    # [existing imports and initial setup]

    # Process unit system
    unit_system = kwargs.get('unit_system')
    if unit_system:
        # Import unit systems
        from pyMAOS import unit_manager
        from unit_manager import SI_UNITS, IMPERIAL_UNITS, METRIC_KN_UNITS, set_unit_system

        # Use the specified unit system for display
        if unit_system == "imperial":
            display_units = IMPERIAL_UNITS.copy()  # Make a copy so we can modify it
            system_name = "Imperial"
            # Add distance unit (feet) for node positions specifically in imperial units
            display_units['distance'] = 'ft'
        elif unit_system == "si":
            display_units = SI_UNITS
            system_name = "SI"
            # Use same unit for distance as length in SI
            display_units['distance'] = display_units['length']
        elif unit_system == "metric_kn":
            display_units = METRIC_KN_UNITS
            system_name = "Metric kN"
            # Use same unit for distance as length in metric_kn
            display_units['distance'] = display_units['length']
        else:
            display_units = self.units.copy()
            # Default to using length unit for distance if not specified
            if 'distance' not in display_units:
                display_units['distance'] = display_units['length']
            system_name = "Current"
    else:
        display_units = self.units.copy()
        # Default to using length unit for distance if not specified
        if 'distance' not in display_units:
            display_units['distance'] = display_units['length']
        system_name = "Current"

    print(f"Using {system_name} units for Excel export")

    def convert_value(value, from_unit, to_unit):
        """Convert a value from one unit to another"""
        try:
            # Handle special case for dimensionless units like radians
            if to_unit in ['rad', 'radian', 'radians']:
                return value

            # Convert using pint
            return Q_(value, from_unit).to(to_unit).magnitude
        except Exception as e:
            print(f"Warning: Could not convert {value} from {from_unit} to {to_unit}: {e}")
            return value

    # Resolve output file path
    output_file = Path(output_file)

    # Get list of all load combinations that have been analyzed
    if loadcombos is None:
        # Find all unique load combos that have results
        all_combos = set()
        for node in self.nodes:
            if hasattr(node, 'displacements'):
                all_combos.update(node.displacements.keys())
        from pyMAOS.loadcombos import LoadCombo
        loadcombos = [LoadCombo(name, {name: 1.0}, [name], False, "CUSTOM") for name in all_combos]

    if not loadcombos:
        raise ValueError("No load combinations specified and no analysis results found.")

    # Extract options
    include_visualization = kwargs.get('include_visualization', True)

    print(f"Exporting analysis results to {output_file}...")

    # Create Excel writer
    with pd.ExcelWriter(output_file, engine='xlsxwriter') as writer:
        workbook = writer.book

        # Create formats
        try:
            header_format = workbook.add_format({
                'bold': True,
                'text_wrap': True,
                'valign': 'top',
                'fg_color': '#D7E4BC',
                'border': 1
            })

            # Create a function to adjust column widths based on headers
            def adjust_column_widths(worksheet, df):
                for idx, col in enumerate(df.columns):
                    # Calculate column width based on header length
                    # Add some padding (1.2 multiplier) to ensure text fits
                    width = max(len(str(col)) * 1.2, 10)  # Minimum width of 10
                    worksheet.set_column(idx, idx, width)

                    # Apply the header format to the header row
                    worksheet.write(0, idx, str(col), header_format)

                # Return the modified worksheet
                return worksheet
        except AttributeError:
            header_format = None

            # Dummy function if header_format creation fails
            def adjust_column_widths(worksheet, df):
                return worksheet

        # 1. Summary sheet
        summary_data = {
            'Parameter': ['Number of Nodes', 'Number of Members', 'Degrees of Freedom',
                          'Number of Restraints', 'Analysis Type'],
            'Value': [self.NJ, self.NM, self.NDOF, self.NR, 'Linear Static']
        }
        # Add load combinations to summary
        for i, combo in enumerate(loadcombos):
            summary_data['Parameter'].append(f"Load Combination {i + 1}")
            summary_data['Value'].append(combo.name)

        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        # After writing a dataframe to Excel:
        worksheet = writer.sheets['Summary']
        adjust_column_widths(worksheet, summary_df)

        # 2. Structure visualization (if requested)
        # Modify _export_results_to_excel to convert coordinates for plotting
        if include_visualization:
            try:
                from pyMAOS.structure2d_plot_with_matplotlib import plot_structure_matplotlib

                # Create temporary lists of nodes with converted coordinates for plotting
                class TempNode:
                    def __init__(self, uid, x, y):
                        self.uid = uid
                        self.x = x
                        self.y = y

                # Convert node coordinates to display units
                temp_nodes = []
                for node in self.nodes:
                    # Convert from meters to the display unit (e.g., inches or feet)
                    x_display = convert_value(node.x, 'm', display_units['distance'])
                    y_display = convert_value(node.y, 'm', display_units['distance'])
                    temp_nodes.append(TempNode(node.uid, x_display, y_display))

                # Create temporary members with converted nodes
                class TempMember:
                    def __init__(self, uid, inode, jnode, member_type, hinges=None):
                        self.uid = uid
                        self.inode = inode
                        self.jnode = jnode
                        self.type = member_type
                        self.hinges = hinges

                temp_members = []
                for member in self.members:
                    # Find the converted i and j nodes
                    i_node = next(n for n in temp_nodes if n.uid == member.inode.uid)
                    j_node = next(n for n in temp_nodes if n.uid == member.jnode.uid)

                    # Create temporary member with hinges if applicable
                    hinges = member.hinges if hasattr(member, 'hinges') else None
                    temp_members.append(TempMember(member.uid, i_node, j_node, member.type, hinges))

                # Use the existing plot function with converted coordinates
                fig, ax = plot_structure_matplotlib(temp_nodes, temp_members)

                # Add units to axis labels
                ax.set_xlabel(f'X ({display_units["distance"]})')
                ax.set_ylabel(f'Y ({display_units["distance"]})')
                ax.set_title(f'Structure Plot ({system_name} Units)')

                # Add visualization to Excel
                worksheet = workbook.add_worksheet('Structure Visualization')

                # Save the figure to a BytesIO object
                imgdata = io.BytesIO()
                fig.savefig(imgdata, format='png', dpi=150, bbox_inches='tight')
                imgdata.seek(0)

                # Insert the image into the worksheet
                worksheet.insert_image('A1', 'structure.png',
                                      {'image_data': imgdata, 'x_scale': 0.8, 'y_scale': 0.8})

                # Close the matplotlib figure to free memory
                plt.close(fig)
                print(f"Added structure visualization in {system_name} units")

            except (ImportError, AttributeError) as e:
                print(f"Warning: Could not create structure visualization: {e}")

        # 3. Units sheet
        units_data = []
        for dimension, unit in display_units.items():
            units_data.append({'Dimension': dimension, 'Unit': unit})
        units_df = pd.DataFrame(units_data)
        units_df.to_excel(writer, sheet_name='Units', index=False)
        # After writing a dataframe to Excel:
        worksheet = writer.sheets['Units']
        adjust_column_widths(worksheet, units_df)

        # Process each load combination
        for combo in loadcombos:
            combo_name = combo.name
            sheet_name = f"Results_{combo_name}"[:31]  # Excel sheet name limit is 31 chars

            # 4. Node information sheet for this load combination
            nodes_data = []
            for node in sorted(self.nodes, key=lambda n: n.uid):
                # Convert coordinates to display units - using 'distance' (feet) instead of 'length'
                x_display = convert_value(node.x, 'm', display_units['distance'])
                y_display = convert_value(node.y, 'm', display_units['distance'])

                node_info = {
                    'Node ID': node.uid,
                    f'X ({display_units["distance"]})': x_display,
                    f'Y ({display_units["distance"]})': y_display,
                    # [rest of node info code remains the same]
                }

                # [keep existing code for displacements and reactions]

                nodes_data.append(node_info)

            nodes_df = pd.DataFrame(nodes_data)
            nodes_df.to_excel(writer, sheet_name=sheet_name, index=False)
            # After writing a dataframe to Excel:
            worksheet = writer.sheets[sheet_name]
            adjust_column_widths(worksheet, nodes_df)

            # Create member forces sheet
            _export_member_forces(self, writer, combo, combo_name, display_units, convert_value)

            # Add fixed end forces sheet for this load combo
            _export_fixed_end_forces(self, writer, combo, combo_name, display_units, convert_value)

        # Add a member properties sheet (common to all load combos)
        _export_member_properties(self, writer, display_units, convert_value)

    print(f"Successfully exported results to {output_file}")
    return str(output_file)

def _export_fixed_end_forces(structure, writer, combo, combo_name, display_units, convert_value):
    """Helper function to export fixed end forces to Excel"""
    import numpy as np
    import pandas as pd

    fef_data = []
    for member in sorted(structure.members, key=lambda m: m.uid):
        # Get fixed end forces if available
        if hasattr(member, 'fixed_end_forces') and combo_name in member.fixed_end_forces:
            fef = member.fixed_end_forces[combo_name]

            # Convert forces to display units
            fef_display = [
                convert_value(fef[0], 'N', display_units['force']),     # i-node Fx
                convert_value(fef[1], 'N', display_units['force']),     # i-node Fy
                convert_value(fef[2], 'N*m', display_units['moment']),  # i-node Mz
                convert_value(fef[3], 'N', display_units['force']),     # j-node Fx
                convert_value(fef[4], 'N', display_units['force']),     # j-node Fy
                convert_value(fef[5], 'N*m', display_units['moment'])   # j-node Mz
            ]

            # i-node fixed end forces
            fef_data.append({
                'Member ID': member.uid,
                'Node': f"{member.inode.uid} (i)",
                f'Fx ({display_units["force"]})': fef_display[0],
                f'Fy ({display_units["force"]})': fef_display[1],
                f'Mz ({display_units["moment"]})': fef_display[2]
            })

            # j-node fixed end forces
            fef_data.append({
                'Member ID': member.uid,
                'Node': f"{member.jnode.uid} (j)",
                f'Fx ({display_units["force"]})': fef_display[3],
                f'Fy ({display_units["force"]})': fef_display[4],
                f'Mz ({display_units["moment"]})': fef_display[5]
            })
        else:
            # If no fixed end forces, add zeros
            fef_data.append({
                'Member ID': member.uid,
                'Node': f"{member.inode.uid} (i)",
                f'Fx ({display_units["force"]})': 0.0,
                f'Fy ({display_units["force"]})': 0.0,
                f'Mz ({display_units["moment"]})': 0.0
            })

            fef_data.append({
                'Member ID': member.uid,
                'Node': f"{member.jnode.uid} (j)",
                f'Fx ({display_units["force"]})': 0.0,
                f'Fy ({display_units["force"]})': 0.0,
                f'Mz ({display_units["moment"]})': 0.0
            })

    # Write fixed end forces to a separate sheet for this load combo
    if fef_data:
        fef_df = pd.DataFrame(fef_data)
        sheet_name = f"FixedEndForces_{combo_name}"[:31]  # Excel sheet name limit is 31 chars
        fef_df.to_excel(writer, sheet_name=sheet_name, index=False)
        # After writing a dataframe to Excel:
        worksheet = writer.sheets[sheet_name]
        adjust_column_widths(worksheet, fef_df)
        print(f"Added Fixed End Forces sheet for {combo_name}")

def _export_member_forces(structure, writer, combo, combo_name, display_units, convert_value):
    """Helper function to export member forces to Excel"""
    import numpy as np
    import pandas as pd
    
    forces_data = []
    for member in sorted(structure.members, key=lambda m: m.uid):
        # Get forces
        global_forces, local_forces = _get_member_forces(member, combo, combo_name)
        
        # Convert forces to display units
        global_forces_display = [
            convert_value(global_forces[0], 'N', display_units['force']),
            convert_value(global_forces[1], 'N', display_units['force']),
            convert_value(global_forces[2], 'N*m', display_units['moment']),
            convert_value(global_forces[3], 'N', display_units['force']),
            convert_value(global_forces[4], 'N', display_units['force']),
            convert_value(global_forces[5], 'N*m', display_units['moment'])
        ]
        
        local_forces_display = [
            convert_value(local_forces[0], 'N', display_units['force']),
            convert_value(local_forces[1], 'N', display_units['force']),
            convert_value(local_forces[2], 'N*m', display_units['moment']),
            convert_value(local_forces[3], 'N', display_units['force']),
            convert_value(local_forces[4], 'N', display_units['force']),
            convert_value(local_forces[5], 'N*m', display_units['moment'])
        ]
        
        # i-node forces
        forces_data.append({
            'Member ID': member.uid,
            'Node': f"{member.inode.uid} (i)",
            'System': 'Global',
            f'Fx ({display_units["force"]})': global_forces_display[0],
            f'Fy ({display_units["force"]})': global_forces_display[1],
            f'Mz ({display_units["moment"]})': global_forces_display[2]
        })
        forces_data.append({
            'Member ID': member.uid,
            'Node': f"{member.inode.uid} (i)",

            'System': 'Local',
            f'Fx ({display_units["force"]})': local_forces_display[0],
            f'Fy ({display_units["force"]})': local_forces_display[1],
            f'Mz ({display_units["moment"]})': local_forces_display[2],
        })
        
        # j-node forces
        forces_data.append({
            'Member ID': member.uid,
            'Node': f"{member.jnode.uid} (j)",
            'System': 'Global',
            f'Fx ({display_units["force"]})': global_forces_display[3],
            f'Fy ({display_units["force"]})': global_forces_display[4],
            f'Mz ({display_units["moment"]})': global_forces_display[5]
        })
        forces_data.append({
            'Member ID': member.uid,
            'Node': f"{member.jnode.uid} (j)",

            'System': 'Local',
            f'Fx ({display_units["force"]})': local_forces_display[3],
            f'Fy ({display_units["force"]})': local_forces_display[4],
            f'Mz ({display_units["moment"]})': local_forces_display[5],
        })
        
    # Write member forces to a separate sheet for this load combo
    forces_df = pd.DataFrame(forces_data)
    sheet_name = f"Forces_{combo_name}"[:31]  # Excel sheet name limit is 31 chars
    forces_df.to_excel(writer, sheet_name=sheet_name, index=False)
    # After writing a dataframe to Excel:
    worksheet = writer.sheets[sheet_name]
    adjust_column_widths(worksheet, forces_df)

def _export_member_properties(structure, writer, display_units, convert_value):
    """Helper function to export member properties to Excel"""
    import pandas as pd
    
    members_data = []
    for member in sorted(structure.members, key=lambda m: m.uid):
        # Convert member properties to display units
        length_display = convert_value(member.length, 'm', display_units['length'])
        e_display = convert_value(member.material.E, 'Pa', display_units['pressure'])
        area_display = convert_value(member.section.Area, 'm^2', f"{display_units['length']}^2")
        ixx_display = convert_value(member.section.Ixx, 'm^4', f"{display_units['length']}^4")
        
        member_info = {
            'Member ID': member.uid,
            'Type': member.type,
            'i-node': member.inode.uid,
            'j-node': member.jnode.uid,
            f'Length ({display_units["length"]})': length_display,
            'Material ID': member.material.uid,
            f'E ({display_units["pressure"]})': e_display,
            'Section ID': member.section.uid,
            f'Area ({display_units["length"]}²)': area_display,
            f'Ixx ({display_units["length"]}⁴)': ixx_display,
        }
        
        # Add hinge information if it's a frame
        if hasattr(member, 'hinges'):
            hinge_info = []
            if member.hinges[0]:
                hinge_info.append('i-node')
            if member.hinges[1]:
                hinge_info.append('j-node')
            member_info['Hinges'] = ', '.join(hinge_info) if hinge_info else 'None'
        
        members_data.append(member_info)
    
    members_df = pd.DataFrame(members_data)
    members_df.to_excel(writer, sheet_name='Member Properties', index=False)
    # After writing a dataframe to Excel:
    worksheet = writer.sheets['Member Properties']
    adjust_column_widths(worksheet, members_df)

def _get_member_forces(member, combo, combo_name):
    """Helper function to get member forces in both global and local coordinates"""
    import numpy as np
    
    # Calculate member forces for this load combination
    if hasattr(member, 'end_forces_global') and combo_name in member.end_forces_global:
        global_forces = np.asarray(member.end_forces_global[combo_name]).flatten()
    else:
        # Calculate forces if not already available
        try:
            global_forces = np.asarray(member.set_end_forces_global(combo)).flatten()
        except:
            global_forces = np.zeros(6)
    
    if hasattr(member, 'end_forces_local') and combo_name in member.end_forces_local:
        local_forces = np.asarray(member.end_forces_local[combo_name]).flatten()
    else:
        try:
            member.Flocal(combo)
            local_forces = np.asarray(member.end_forces_local[combo_name]).flatten()
        except:
            local_forces = np.zeros(6)
    
    return global_forces, local_forces
