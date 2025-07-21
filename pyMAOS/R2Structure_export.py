def export_results_to_excel(self, output_file, loadcombos=None, **kwargs):
    """
    Export structural analysis results to Excel format with multiple sheets including visualization
    
    Parameters
    ----------
    output_file : str or Path
        Path for the output Excel file
    loadcombos : list of LoadCombo objects, optional
        List of load combinations to include in the export (if None, uses all analyzed load combinations)
    **kwargs : dict
        Additional options:
        - include_visualization : bool, default True
            Whether to include structure visualization sheet
        - unit_system : str, default None
            The unit system to use ("imperial", "si", "metric_kn")
            If None, uses the current unit system from the model
        - scaling : dict, optional
            Scaling factors for visualization
    """
    # Check for required packages
    try:
        import pandas as pd
        import matplotlib.pyplot as plt
        import numpy as np
        import io
        from pathlib import Path
        from pint import UnitRegistry
    except ImportError as e:
        raise ImportError(f"Required package not available for Excel export: {e}")
    
    # Create unit registry for conversions
    ureg = UnitRegistry()
    Q_ = ureg.Quantity
    
    # Process unit system
    unit_system = kwargs.get('unit_system')
    if unit_system:
        # Import unit systems
        from pyMAOS.units_mod import SI_UNITS, IMPERIAL_UNITS, METRIC_KN_UNITS, set_unit_system
        
        # Use the specified unit system for display
        if unit_system == "imperial":
            display_units = IMPERIAL_UNITS
            system_name = "Imperial"
        elif unit_system == "si":
            display_units = SI_UNITS
            system_name = "SI"
        elif unit_system == "metric_kn":
            display_units = METRIC_KN_UNITS
            system_name = "Metric kN"
        else:
            display_units = self.units
            system_name = "Current"
    else:
        display_units = self.units
        system_name = "Current"
        
    print(f"Using {system_name} units for Excel export")

    # Utility function for unit conversion
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
        except AttributeError:
            header_format = None
        
        # 1. Summary sheet
        summary_data = {
            'Parameter': ['Number of Nodes', 'Number of Members', 'Degrees of Freedom', 
                         'Number of Restraints', 'Analysis Type'],
            'Value': [self.NJ, self.NM, self.NDOF, self.NR, 'Linear Static']
        }
        # Add load combinations to summary
        for i, combo in enumerate(loadcombos):
            summary_data['Parameter'].append(f"Load Combination {i+1}")
            summary_data['Value'].append(combo.name)
            
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name='Summary', index=False)
        
        # 2. Structure visualization (if requested)
        if include_visualization:
            try:
                from pyMAOS.plot_with_matplotlib import plot_structure_matplotlib
                
                # Use the existing plot function
                fig, ax = plot_structure_matplotlib(self.nodes, self.members)
                
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
                
            except (ImportError, AttributeError) as e:
                print(f"Warning: Could not create structure visualization: {e}")
        
        # 3. Units sheet
        units_data = []
        for dimension, unit in display_units.items():
            units_data.append({'Dimension': dimension, 'Unit': unit})
        units_df = pd.DataFrame(units_data)
        units_df.to_excel(writer, sheet_name='Units', index=False)
        
        # Process each load combination
        for combo in loadcombos:
            combo_name = combo.name
            sheet_name = f"Results_{combo_name}"[:31]  # Excel sheet name limit is 31 chars
            
            # 4. Node information sheet for this load combination
            nodes_data = []
            for node in sorted(self.nodes, key=lambda n: n.uid):
                # Convert coordinates to display units
                x_display = convert_value(node.x, 'm', display_units['length'])
                y_display = convert_value(node.y, 'm', display_units['length'])
                
                node_info = {
                    'Node ID': node.uid,
                    f'X ({display_units["length"]})': x_display,
                    f'Y ({display_units["length"]})': y_display,
                    'Restrained X': node.restraints[0],
                    'Restrained Y': node.restraints[1],
                    'Restrained Z': node.restraints[2],
                }
                
                # Add displacements if available
                if hasattr(node, 'displacements') and combo_name in node.displacements:
                    disp = node.displacements[combo_name]
                    # Convert to display units
                    disp_x = convert_value(disp[0], 'm', display_units['length'])
                    disp_y = convert_value(disp[1], 'm', display_units['length']) 
                    rot_z = disp[2]  # Radians are dimensionless
                    
                    node_info.update({
                        f'Displacement X ({display_units["length"]})': disp_x,
                        f'Displacement Y ({display_units["length"]})': disp_y,
                        'Rotation Z (rad)': rot_z,
                    })
                else:
                    node_info.update({
                        f'Displacement X ({display_units["length"]})': 0.0,
                        f'Displacement Y ({display_units["length"]})': 0.0,
                        'Rotation Z (rad)': 0.0,
                    })
                
                # Add reactions if available
                if hasattr(node, 'reactions') and combo_name in node.reactions:
                    reaction = node.reactions[combo_name]
                    # Convert to display units
                    rx = convert_value(reaction[0], 'N', display_units['force'])
                    ry = convert_value(reaction[1], 'N', display_units['force'])
                    mz = convert_value(reaction[2], 'N*m', display_units['moment'])
                    
                    node_info.update({
                        f'Reaction X ({display_units["force"]})': rx,
                        f'Reaction Y ({display_units["force"]})': ry,
                        f'Moment Z ({display_units["moment"]})': mz,
                    })
                else:
                    node_info.update({
                        f'Reaction X ({display_units["force"]})': 0.0,
                        f'Reaction Y ({display_units["force"]})': 0.0,
                        f'Moment Z ({display_units["moment"]})': 0.0,
                    })
                
                nodes_data.append(node_info)
            
            nodes_df = pd.DataFrame(nodes_data)
            nodes_df.to_excel(writer, sheet_name=sheet_name, index=False)
            
            # Create member forces sheet
            _export_member_forces(self, writer, combo, combo_name, display_units, convert_value)
        
        # Add a member properties sheet (common to all load combos)
        _export_member_properties(self, writer, display_units, convert_value)
    
    print(f"Successfully exported results to {output_file}")
    return str(output_file)

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
        
        # i-node global forces
        forces_data.append({
            'Member ID': member.uid,
            'Node': f"{member.inode.uid} (i)",
            f'Fx ({display_units["force"]})': global_forces_display[0],
            f'Fy ({display_units["force"]})': global_forces_display[1],
            f'Mz ({display_units["moment"]})': global_forces_display[2],
            'System': 'Global'
        })
        
        # j-node global forces
        forces_data.append({
            'Member ID': member.uid,
            'Node': f"{member.jnode.uid} (j)",
            f'Fx ({display_units["force"]})': global_forces_display[3],
            f'Fy ({display_units["force"]})': global_forces_display[4],
            f'Mz ({display_units["moment"]})': global_forces_display[5],
            'System': 'Global'
        })
        
        # i-node local forces
        forces_data.append({
            'Member ID': member.uid,
            'Node': f"{member.inode.uid} (i)",
            f'Fx ({display_units["force"]})': local_forces_display[0],
            f'Fy ({display_units["force"]})': local_forces_display[1],
            f'Mz ({display_units["moment"]})': local_forces_display[2],
            'System': 'Local'
        })
        
        # j-node local forces
        forces_data.append({
            'Member ID': member.uid,
            'Node': f"{member.jnode.uid} (j)",
            f'Fx ({display_units["force"]})': local_forces_display[3],
            f'Fy ({display_units["force"]})': local_forces_display[4],
            f'Mz ({display_units["moment"]})': local_forces_display[5],
            'System': 'Local'
        })
    
    # Write member forces to a separate sheet for this load combo
    forces_df = pd.DataFrame(forces_data)
    sheet_name = f"Forces_{combo_name}"[:31]  # Excel sheet name limit is 31 chars
    forces_df.to_excel(writer, sheet_name=sheet_name, index=False)

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

def _get_member_forces(member, combo, combo_name):
    """Helper function to get member forces in both global and local coordinates"""
    import numpy as np
    
    # Calculate member forces for this load combination
    if hasattr(member, 'end_forces_global') and combo_name in member.end_forces_global:
        global_forces = np.asarray(member.end_forces_global[combo_name]).flatten()
    else:
        # Calculate forces if not already available
        try:
            global_forces = np.asarray(member.Fglobal(combo)).flatten()
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
