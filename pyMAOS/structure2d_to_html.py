def generate_html_report(self, loadcombos=None, output_html="results.html"):
    """
    Generate a comprehensive interactive HTML report for structural analysis results using Plotly.

    Parameters
    ----------
    loadcombos : list, optional
        List of load combinations to include in the report
    output_html : str, optional
        Path to save the HTML report

    Returns
    -------
    str
        Path to the created HTML file
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import numpy as np
    import os
    import plotly.io as pio
    import pandas as pd
    from plotly.colors import qualitative

    print(f"Generating comprehensive interactive HTML report: {output_html}")

    # Use all load combos if none specified
    if loadcombos is None:
        # Determine available load combos from node displacements
        if self.nodes and hasattr(self.nodes[0], 'displacements') and self.nodes[0].displacements:
            available_combos = list(self.nodes[0].displacements.keys())
            loadcombos = available_combos
        else:
            print("Warning: No load combinations found in results.")
            return None

    # Extract values safely (handling Pint quantities)
    def extract_value(val):
        try:
            if hasattr(val, 'magnitude'):  # Check if it's a Pint quantity
                return float(val.magnitude)
            return float(val)
        except Exception as e:
            print(f"Error extracting value: {e}")
            return 0.0
    
    def extract_unit(val):
        """Extract unit string from a quantity"""
        try:
            if hasattr(val, 'units'):
                return str(val.units)
            return ""
        except Exception:
            return ""
    
    # Get display units from the first few quantities we encounter
    display_units = {}
    
    # Try to determine units from first node coordinates
    if self.nodes:
        if hasattr(self.nodes[0].x, 'units'):
            display_units['length'] = str(self.nodes[0].x.units)
        
    # Try to determine force units from first member reaction
    if self.members and hasattr(self.members[0], 'end_forces_local'):
        for combo_name, forces in self.members[0].end_forces_local.items():
            if forces is not None and len(forces) > 0:
                if hasattr(forces[0], 'units'):
                    display_units['force'] = str(forces[0].units)
                    break
    
    # Set default units if not found
    if 'length' not in display_units:
        display_units['length'] = 'm'
    if 'force' not in display_units:
        display_units['force'] = 'N'
    if 'moment' not in display_units:
        display_units['moment'] = f"{display_units['force']}⋅{display_units['length']}"
        
    print(f"Using display units: {display_units}")
    
    # Create HTML with multiple tabs using Plotly's Dash components but rendered as static HTML
    # First, create a list to hold all figures
    figures = []
    
    # 1. Create summary information table
    summary_fig = go.Figure(data=[go.Table(
        header=dict(
            values=['Parameter', 'Value'],
            fill_color='lightblue',
            align='left',
            font=dict(size=14)
        ),
        cells=dict(
            values=[
                ['Number of Nodes', 'Number of Members', 'Degrees of Freedom', 
                 'Number of Restraints', 'Analysis Type', 'Load Combinations'],
                [len(self.nodes), len(self.members), self.NDOF, 
                 sum(sum(node.restraints) for node in self.nodes), 'Linear Static',
                 ', '.join(combo for combo in loadcombos)]
            ],
            fill_color='lavender',
            align='left',
            font=dict(size=12)
        )
    )])
    
    summary_fig.update_layout(
        title_text="Analysis Summary",
        height=300,
        margin=dict(l=0, r=0, b=0, t=30),
    )
    
    figures.append(summary_fig)
    
    # 2. Create structure visualization with loads
    structure_fig = go.Figure()
    
    # Add members
    for member in self.members:
        structure_fig.add_trace(
            go.Scatter(
                x=[extract_value(member.inode.x), extract_value(member.jnode.x)],
                y=[extract_value(member.inode.y), extract_value(member.jnode.y)],
                mode='lines',
                line=dict(color='blue', width=3),
                name=f"Member {member.uid}"
            )
        )
    
    # Add nodes with hover text showing restraints
    for node in self.nodes:
        hover_text = (f"Node {node.uid}<br>"
                     f"Restraints: {node.restraints}<br>"
                     f"X: {extract_value(node.x)} {display_units['length']}<br>"
                     f"Y: {extract_value(node.y)} {display_units['length']}")

        # Use different markers for restrained nodes
        marker_symbol = 'circle'
        marker_color = 'blue'
        if sum(node.restraints) > 0:
            marker_symbol = 'square'
            marker_color = 'red'

        structure_fig.add_trace(
            go.Scatter(
                x=[extract_value(node.x)],
                y=[extract_value(node.y)],
                mode='markers',
                marker=dict(size=10, symbol=marker_symbol, color=marker_color),
                name=f"Node {node.uid}",
                text=hover_text,
                hoverinfo='text'
            )
        )
        
        # Add nodal loads (if any)
        for combo in loadcombos:
            if hasattr(node, 'loads') and combo in node.loads:
                load = node.loads[combo]
                fx = extract_value(load[0])
                fy = extract_value(load[1])
                mz = extract_value(load[2])
                
                # Only show loads that are not zero
                load_magnitude = np.sqrt(fx*fx + fy*fy)
                if load_magnitude > 0:
                    # Scale arrow for visibility
                    scale = min(5.0, max(0.5, 20.0 / load_magnitude)) if load_magnitude > 0 else 1.0
                    
                    # Add arrow for force
                    arrow_x = extract_value(node.x)
                    arrow_y = extract_value(node.y)
                    arrow_dx = -fx * scale  # Negative because loads are in opposite direction
                    arrow_dy = -fy * scale
                    
                    structure_fig.add_trace(
                        go.Scatter(
                            x=[arrow_x, arrow_x + arrow_dx],
                            y=[arrow_y, arrow_y + arrow_dy],
                            mode='lines+markers',
                            line=dict(color='red', width=2),
                            marker=dict(size=[0, 8], symbol='arrow', angle=np.degrees(np.arctan2(arrow_dy, arrow_dx))),
                            name=f"Load {combo} at Node {node.uid}",
                            text=f"Fx={fx:.2f} {display_units['force']}<br>Fy={fy:.2f} {display_units['force']}",
                            hoverinfo='text'
                        )
                    )
                
                # Add curved arrow for moment (if not zero)
                if abs(mz) > 1e-8:
                    # Create a small circle to represent moment
                    theta = np.linspace(0, 2*np.pi, 20)
                    radius = 0.05 * max([extract_value(m.length) for m in self.members if hasattr(m, 'length')])
                    
                    # Direction depends on sign of moment
                    if mz > 0:  # Counter-clockwise
                        structure_fig.add_trace(
                            go.Scatter(
                                x=extract_value(node.x) + radius * np.cos(theta),
                                y=extract_value(node.y) + radius * np.sin(theta),
                                mode='lines',
                                line=dict(color='red', width=2),
                                name=f"Moment {combo} at Node {node.uid}",
                                text=f"Mz={mz:.2f} {display_units['moment']}",
                                hoverinfo='text'
                            )
                        )
                        # Add arrowhead
                        structure_fig.add_trace(
                            go.Scatter(
                                x=[extract_value(node.x) + radius, extract_value(node.x) + radius*0.8, 
                                   extract_value(node.x) + radius*0.8],
                                y=[extract_value(node.y), extract_value(node.y) + radius*0.1, 
                                   extract_value(node.y) - radius*0.1],
                                mode='lines',
                                line=dict(color='red', width=2),
                                showlegend=False
                            )
                        )
                    else:  # Clockwise
                        structure_fig.add_trace(
                            go.Scatter(
                                x=extract_value(node.x) + radius * np.cos(theta),
                                y=extract_value(node.y) + radius * np.sin(theta),
                                mode='lines',
                                line=dict(color='red', width=2),
                                name=f"Moment {combo} at Node {node.uid}",
                                text=f"Mz={mz:.2f} {display_units['moment']}",
                                hoverinfo='text'
                            )
                        )
                        # Add arrowhead
                        structure_fig.add_trace(
                            go.Scatter(
                                x=[extract_value(node.x) - radius, extract_value(node.x) - radius*0.8, 
                                   extract_value(node.x) - radius*0.8],
                                y=[extract_value(node.y), extract_value(node.y) + radius*0.1, 
                                   extract_value(node.y) - radius*0.1],
                                mode='lines',
                                line=dict(color='red', width=2),
                                showlegend=False
                            )
                        )
    
    # Add distributed loads on members
    for member in self.members:
        if hasattr(member, 'loads'):
            for load in member.loads:
                # Handle different types of loads
                if hasattr(load, 'kind') and load.kind == 'DISTRIBUTED':
                    # Draw distributed load
                    length = extract_value(member.length)
                    num_arrows = 5  # Number of arrows to represent distributed load
                    
                    # Get local coordinates
                    cos_theta = (extract_value(member.jnode.x) - extract_value(member.inode.x)) / length
                    sin_theta = (extract_value(member.jnode.y) - extract_value(member.inode.y)) / length
                    
                    # For each arrow position
                    for i in range(num_arrows):
                        x_local = length * (i + 0.5) / num_arrows
                        
                        # Get load magnitude at this position (assuming w1 and w2 attributes for linear load)
                        if hasattr(load, 'w1') and hasattr(load, 'w2'):
                            w1 = extract_value(load.w1)
                            w2 = extract_value(load.w2)
                            w_x = w1 + (w2 - w1) * x_local / length
                        else:
                            w_x = extract_value(load.p) if hasattr(load, 'p') else 1.0
                        
                        # Position along member
                        x_pos = extract_value(member.inode.x) + x_local * cos_theta
                        y_pos = extract_value(member.inode.y) + x_local * sin_theta
                        
                        # Arrow direction (perpendicular to member)
                        scale = 0.1 * length  # Scale for visibility
                        
                        structure_fig.add_trace(
                            go.Scatter(
                                x=[x_pos, x_pos - w_x * scale * sin_theta],
                                y=[y_pos, y_pos + w_x * scale * cos_theta],
                                mode='lines+markers',
                                line=dict(color='purple', width=2),
                                marker=dict(size=[0, 8], symbol='arrow'),
                                name=f"Distributed Load on Member {member.uid}",
                                showlegend=(i == 0)
                            )
                        )
    
    structure_fig.update_layout(
        title_text="Structure Geometry and Loads",
        xaxis=dict(
            title=f"X ({display_units['length']})",
            constrain="domain",
            scaleanchor="y",
            scaleratio=1,
        ),
        yaxis=dict(
            title=f"Y ({display_units['length']})"
        ),
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor="rgba(255, 255, 255, 0.8)"
        ),
        height=700,
        margin=dict(l=0, r=0, b=0, t=50),
    )
    
    figures.append(structure_fig)
    
    # 3. Create deformed shape for all load combinations
    for combo_idx, combo in enumerate(loadcombos):
        deformed_fig = go.Figure()
        
        # Add original structure (undeformed)
        for member in self.members:
            deformed_fig.add_trace(
                go.Scatter(
                    x=[extract_value(member.inode.x), extract_value(member.jnode.x)],
                    y=[extract_value(member.inode.y), extract_value(member.jnode.y)],
                    mode='lines',
                    line=dict(color='blue', width=2, dash='dot'),
                    name="Undeformed Shape"
                )
            )
        
        # Get max displacement for scaling
        max_disp = 0.0
        for node in self.nodes:
            if hasattr(node, 'displacements') and combo in node.displacements:
                ux = extract_value(node.displacements[combo][0])
                uy = extract_value(node.displacements[combo][1])
                disp_magnitude = np.sqrt(ux**2 + uy**2)
                max_disp = max(max_disp, disp_magnitude)
        
        # Apply scaling factor for better visualization
        scale_factor = 20.0  # Default
        if max_disp > 0:
            avg_member_length = np.mean([extract_value(m.length) for m in self.members])
            scale_factor = 0.2 * avg_member_length / max_disp
            scale_factor = min(scale_factor, 50.0)  # Limit scaling
        
        print(f"Load combo: {combo}, Max displacement: {max_disp:.6f}, Scale factor: {scale_factor}")
        
        # Draw deformed shape
        for member_idx, member in enumerate(self.members):
            i_node = member.inode
            j_node = member.jnode
            
            # Get original coordinates
            x1, y1 = extract_value(i_node.x), extract_value(i_node.y)
            x2, y2 = extract_value(j_node.x), extract_value(j_node.y)
            
            # Get displacements if available
            ux1, uy1, ux2, uy2 = 0, 0, 0, 0
            
            if hasattr(i_node, 'displacements') and combo in i_node.displacements:
                ux1 = extract_value(i_node.displacements[combo][0])
                uy1 = extract_value(i_node.displacements[combo][1])
            
            if hasattr(j_node, 'displacements') and combo in j_node.displacements:
                ux2 = extract_value(j_node.displacements[combo][0])
                uy2 = extract_value(j_node.displacements[combo][1])
            
            # For more accurate representation, use the member displacement function if available
            if hasattr(member, 'displacement_functions') and combo in member.displacement_functions:
                try:
                    disp_function = member.displacement_functions[combo]
                    
                    # Generate multiple points along member
                    num_points = 20
                    x_local = np.linspace(0, extract_value(member.length), num_points)
                    
                    # Calculate displacement at each point
                    x_global = []
                    y_global = []
                    
                    for x in x_local:
                        # Get member local coordinates
                        cos_theta = (x2 - x1) / extract_value(member.length)
                        sin_theta = (y2 - y1) / extract_value(member.length)
                        
                        # Position along undeformed member
                        x_pos = x1 + x * cos_theta
                        y_pos = y1 + x * sin_theta
                        
                        # Add displacement
                        try:
                            ux = extract_value(disp_function['ux'](x)) if 'ux' in disp_function else 0
                            uy = extract_value(disp_function['uy'](x)) if 'uy' in disp_function else 0
                        except Exception as e:
                            print(f"Error evaluating displacement at x={x}: {e}")
                            ux, uy = 0, 0
                        
                        x_pos += ux * scale_factor
                        y_pos += uy * scale_factor
                        
                        x_global.append(x_pos)
                        y_global.append(y_pos)
                    
                    # Draw the deformed member as a smooth curve
                    deformed_fig.add_trace(
                        go.Scatter(
                            x=x_global,
                            y=y_global,
                            mode='lines',
                            line=dict(color='red', width=3),
                            name=f"Deformed Member {member.uid}"
                        )
                    )
                
                except Exception as e:
                    print(f"Error generating displacement curve for member {member.uid}: {e}")
                    # Fall back to linear deformation
                    x1_def = x1 + ux1 * scale_factor
                    y1_def = y1 + uy1 * scale_factor
                    x2_def = x2 + ux2 * scale_factor
                    y2_def = y2 + uy2 * scale_factor
                    
                    deformed_fig.add_trace(
                        go.Scatter(
                            x=[x1_def, x2_def],
                            y=[y1_def, y2_def],
                            mode='lines',
                            line=dict(color='red', width=3),
                            name=f"Deformed Member {member.uid}"
                        )
                    )
            else:
                # Simple linear deformation
                x1_def = x1 + ux1 * scale_factor
                y1_def = y1 + uy1 * scale_factor
                x2_def = x2 + ux2 * scale_factor
                y2_def = y2 + uy2 * scale_factor
                
                deformed_fig.add_trace(
                    go.Scatter(
                        x=[x1_def, x2_def],
                        y=[y1_def, y2_def],
                        mode='lines',
                        line=dict(color='red', width=3),
                        name=f"Deformed Member {member.uid}"
                    )
                )
        
        # Add deformed node positions with displacement text
        for node in self.nodes:
            if hasattr(node, 'displacements') and combo in node.displacements:
                ux = extract_value(node.displacements[combo][0])
                uy = extract_value(node.displacements[combo][1])
                rz = extract_value(node.displacements[combo][2])
                
                # Calculate deformed position
                x_def = extract_value(node.x) + ux * scale_factor
                y_def = extract_value(node.y) + uy * scale_factor
                
                hover_text = (f"Node {node.uid}<br>"
                             f"Ux: {ux:.4e} {display_units['length']}<br>"
                             f"Uy: {uy:.4e} {display_units['length']}<br>"
                             f"Rz: {rz:.4e} rad")
                
                deformed_fig.add_trace(
                    go.Scatter(
                        x=[x_def],
                        y=[y_def],
                        mode='markers',
                        marker=dict(size=8, color='red'),
                        name=f"Deformed Node {node.uid}",
                        text=hover_text,
                        hoverinfo='text'
                    )
                )
        
        deformed_fig.update_layout(
            title_text=f"Deformed Shape - {combo} (Scale: {scale_factor:.1f}X)",
            xaxis=dict(
                title=f"X ({display_units['length']})",
                constrain="domain",
                scaleanchor="y",
                scaleratio=1,
            ),
            yaxis=dict(
                title=f"Y ({display_units['length']})"
            ),
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(255, 255, 255, 0.8)"
            ),
            height=600,
            margin=dict(l=0, r=0, b=0, t=50),
        )
        
        figures.append(deformed_fig)
    
    # 4. Create internal force diagrams (moment, shear, axial)
    for combo_idx, combo in enumerate(loadcombos):
        # a. Bending Moment Diagram
        moment_fig = go.Figure()
        
        # Add reference structure
        for member in self.members:
            moment_fig.add_trace(
                go.Scatter(
                    x=[extract_value(member.inode.x), extract_value(member.jnode.x)],
                    y=[extract_value(member.inode.y), extract_value(member.jnode.y)],
                    mode='lines',
                    line=dict(color='gray', width=1, dash='dot'),
                    name="Structure"
                )
            )
        
        # Plot moment diagram for each member
        for member_idx, member in enumerate(self.members):
            if not hasattr(member, 'Mz') or combo not in member.Mz:
                continue
            
            length = extract_value(member.length)
            num_points = 31
            
            # Create x positions along member
            x_local = np.linspace(0, length, num_points)
            
            # Get moment values at each position
            moment_values = []
            for x in x_local:
                try:
                    m = extract_value(member.Mz[combo].evaluate(x))
                    moment_values.append(m)
                except Exception as e:
                    print(f"Error evaluating moment at x={x}: {e}")
                    moment_values.append(0)
            
            # Scale for better visualization
            max_moment = max(abs(np.array(moment_values))) if moment_values else 0
            if max_moment > 0:
                moment_scale = 0.15 * length / max_moment
            else:
                moment_scale = 1.0
            
            # Get the global coordinates
            cos_theta = (extract_value(member.jnode.x) - extract_value(member.inode.x)) / length
            sin_theta = (extract_value(member.jnode.y) - extract_value(member.inode.y)) / length
            
            x_global = []
            y_global = []
            hover_texts = []
            
            for i, x in enumerate(x_local):
                # Position along the member in global coordinates
                x_pos = extract_value(member.inode.x) + x * cos_theta
                y_pos = extract_value(member.inode.y) + x * sin_theta
                
                # Add moment value perpendicular to the member
                m = moment_values[i] * moment_scale
                x_pos -= m * sin_theta
                y_pos += m * cos_theta
                
                x_global.append(x_pos)
                y_global.append(y_pos)
                hover_texts.append(f"x: {x:.2f} {display_units['length']}<br>Mz: {moment_values[i]:.2f} {display_units['moment']}")
            
            # Plot the moment diagram
            moment_fig.add_trace(
                go.Scatter(
                    x=x_global,
                    y=y_global,
                    mode='lines',
                    line=dict(color=qualitative.Plotly[member_idx % len(qualitative.Plotly)], width=2),
                    name=f"Member {member.uid}",
                    text=hover_texts,
                    hoverinfo='text'
                )
            )
            
            # Add value labels at ends
            moment_fig.add_trace(
                go.Scatter(
                    x=[x_global[0], x_global[-1]],
                    y=[y_global[0], y_global[-1]],
                    mode='markers+text',
                    marker=dict(size=5, color=qualitative.Plotly[member_idx % len(qualitative.Plotly)]),
                    text=[f"{moment_values[0]:.2f}", f"{moment_values[-1]:.2f}"],
                    textposition=["bottom center", "bottom center"],
                    showlegend=False
                )
            )
        
        moment_fig.update_layout(
            title_text=f"Bending Moment Diagram - {combo}",
            xaxis=dict(
                title=f"X ({display_units['length']})",
                constrain="domain",
                scaleanchor="y",
                scaleratio=1,
            ),
            yaxis=dict(
                title=f"Y ({display_units['length']})"
            ),
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(255, 255, 255, 0.8)"
            ),
            height=600,
            margin=dict(l=0, r=0, b=0, t=50),
        )
        
        figures.append(moment_fig)
        
        # b. Shear Force Diagram
        shear_fig = go.Figure()
        
        # Add reference structure
        for member in self.members:
            shear_fig.add_trace(
                go.Scatter(
                    x=[extract_value(member.inode.x), extract_value(member.jnode.x)],
                    y=[extract_value(member.inode.y), extract_value(member.jnode.y)],
                    mode='lines',
                    line=dict(color='gray', width=1, dash='dot'),
                    name="Structure"
                )
            )
        
        # Plot shear diagram for each member
        for member_idx, member in enumerate(self.members):
            if not hasattr(member, 'Vy') or combo not in member.Vy:
                continue
            
            length = extract_value(member.length)
            num_points = 31
            
            # Create x positions along member
            x_local = np.linspace(0, length, num_points)
            
            # Get shear values at each position
            shear_values = []
            for x in x_local:
                try:
                    v = extract_value(member.Vy[combo].evaluate(x))
                    shear_values.append(v)
                except Exception as e:
                    print(f"Error evaluating shear at x={x}: {e}")
                    shear_values.append(0)
            
            # Scale for better visualization
            max_shear = max(abs(np.array(shear_values))) if shear_values else 0
            if max_shear > 0:
                shear_scale = 0.15 * length / max_shear
            else:
                shear_scale = 1.0
            
            # Get the global coordinates
            cos_theta = (extract_value(member.jnode.x) - extract_value(member.inode.x)) / length
            sin_theta = (extract_value(member.jnode.y) - extract_value(member.inode.y)) / length
            
            x_global = []
            y_global = []
            hover_texts = []
            
            for i, x in enumerate(x_local):
                # Position along the member in global coordinates
                x_pos = extract_value(member.inode.x) + x * cos_theta
                y_pos = extract_value(member.inode.y) + x * sin_theta
                
                # Add shear value perpendicular to the member
                v = shear_values[i] * shear_scale
                x_pos -= v * sin_theta
                y_pos += v * cos_theta
                
                x_global.append(x_pos)
                y_global.append(y_pos)
                hover_texts.append(f"x: {x:.2f} {display_units['length']}<br>Vy: {shear_values[i]:.2f} {display_units['force']}")
            
            # Plot the shear diagram
            shear_fig.add_trace(
                go.Scatter(
                    x=x_global,
                    y=y_global,
                    mode='lines',
                    line=dict(color=qualitative.Plotly[member_idx % len(qualitative.Plotly)], width=2),
                    name=f"Member {member.uid}",
                    text=hover_texts,
                    hoverinfo='text'
                )
            )
            
            # Add value labels at ends
            shear_fig.add_trace(
                go.Scatter(
                    x=[x_global[0], x_global[-1]],
                    y=[y_global[0], y_global[-1]],
                    mode='markers+text',
                    marker=dict(size=5, color=qualitative.Plotly[member_idx % len(qualitative.Plotly)]),
                    text=[f"{shear_values[0]:.2f}", f"{shear_values[-1]:.2f}"],
                    textposition=["bottom center", "bottom center"],
                    showlegend=False
                )
            )
        
        shear_fig.update_layout(
            title_text=f"Shear Force Diagram - {combo}",
            xaxis=dict(
                title=f"X ({display_units['length']})",
                constrain="domain",
                scaleanchor="y",
                scaleratio=1,
            ),
            yaxis=dict(
                title=f"Y ({display_units['length']})"
            ),
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(255, 255, 255, 0.8)"
            ),
            height=600,
            margin=dict(l=0, r=0, b=0, t=50),
        )
        
        figures.append(shear_fig)
        
        # c. Axial Force Diagram
        axial_fig = go.Figure()
        
        # Add reference structure
        for member in self.members:
            axial_fig.add_trace(
                go.Scatter(
                    x=[extract_value(member.inode.x), extract_value(member.jnode.x)],
                    y=[extract_value(member.inode.y), extract_value(member.jnode.y)],
                    mode='lines',
                    line=dict(color='gray', width=1, dash='dot'),
                    name="Structure"
                )
            )
        
        # Plot axial diagram for each member
        for member_idx, member in enumerate(self.members):
            if not hasattr(member, 'N') or combo not in member.N:
                continue
            
            length = extract_value(member.length)
            num_points = 31
            
            # Create x positions along member
            x_local = np.linspace(0, length, num_points)
            
            # Get axial values at each position
            axial_values = []
            for x in x_local:
                try:
                    n = extract_value(member.N[combo].evaluate(x))
                    axial_values.append(n)
                except Exception as e:
                    print(f"Error evaluating axial force at x={x}: {e}")
                    axial_values.append(0)
            
            # Scale for better visualization (smaller scale for axial)
            max_axial = max(abs(np.array(axial_values))) if axial_values else 0
            if max_axial > 0:
                axial_scale = 0.05 * length / max_axial
            else:
                axial_scale = 1.0
            
            # Get member orientation
            cos_theta = (extract_value(member.jnode.x) - extract_value(member.inode.x)) / length
            sin_theta = (extract_value(member.jnode.y) - extract_value(member.inode.y)) / length
            
            # Lines parallel to the member
            x_global = []
            y_global = []
            hover_texts = []
            
            for i, x in enumerate(x_local):
                # Position along the member in global coordinates
                x_pos = extract_value(member.inode.x) + x * cos_theta
                y_pos = extract_value(member.inode.y) + x * sin_theta
                
                # Add axial value perpendicular to the member (rotated 90°)
                n = axial_values[i] * axial_scale
                # For axial force, plot offset perpendicular to member direction
                x_pos += n * sin_theta  # Note: Perpendicular direction
                y_pos -= n * cos_theta  # Note: Perpendicular direction
                
                x_global.append(x_pos)
                y_global.append(y_pos)
                hover_texts.append(f"x: {x:.2f} {display_units['length']}<br>N: {axial_values[i]:.2f} {display_units['force']}")
            
            # Plot the axial diagram
            axial_fig.add_trace(
                go.Scatter(
                    x=x_global,
                    y=y_global,
                    mode='lines',
                    line=dict(color=qualitative.Plotly[member_idx % len(qualitative.Plotly)], width=2),
                    name=f"Member {member.uid}",
                    text=hover_texts,
                    hoverinfo='text'
                )
            )
            
            # Add value labels at ends
            axial_fig.add_trace(
                go.Scatter(
                    x=[x_global[0], x_global[-1]],
                    y=[y_global[0], y_global[-1]],
                    mode='markers+text',
                    marker=dict(size=5, color=qualitative.Plotly[member_idx % len(qualitative.Plotly)]),
                    text=[f"{axial_values[0]:.2f}", f"{axial_values[-1]:.2f}"],
                    textposition=["bottom center", "bottom center"],
                    showlegend=False
                )
            )
        
        axial_fig.update_layout(
            title_text=f"Axial Force Diagram - {combo}",
            xaxis=dict(
                title=f"X ({display_units['length']})",
                constrain="domain",
                scaleanchor="y",
                scaleratio=1,
            ),
            yaxis=dict(
                title=f"Y ({display_units['length']})"
            ),
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                bgcolor="rgba(255, 255, 255, 0.8)"
            ),
            height=600,
            margin=dict(l=0, r=0, b=0, t=50),
        )
        
        figures.append(axial_fig)
    
    # 5. Tabular results (Node displacements, reactions, member forces)
    # a. Node displacements table
    for combo in loadcombos:
        disp_headers = ["Node ID", f"Ux ({display_units['length']})", 
                        f"Uy ({display_units['length']})", "Rz (rad)"]
        disp_data = [[], [], [], []]
        
        for node in sorted(self.nodes, key=lambda n: n.uid):
            if hasattr(node, 'displacements') and combo in node.displacements:
                disp = node.displacements[combo]
                disp_data[0].append(node.uid)
                disp_data[1].append(f"{extract_value(disp[0]):.6e}")
                disp_data[2].append(f"{extract_value(disp[1]):.6e}")
                disp_data[3].append(f"{extract_value(disp[2]):.6e}")
        
        disp_table = go.Figure(data=[go.Table(
            header=dict(
                values=disp_headers,
                fill_color='lightblue',
                align='center',
                font=dict(size=12)
            ),
            cells=dict(
                values=disp_data,
                fill_color='lavender',
                align='right',
                font=dict(size=11)
            )
        )])
        
        disp_table.update_layout(
            title_text=f"Node Displacements - {combo}",
            height=400,
            margin=dict(l=0, r=0, b=0, t=30),
        )
        
        figures.append(disp_table)
    
    # b. Reaction forces table
    for combo in loadcombos:
        react_headers = ["Node ID", f"Rx ({display_units['force']})", 
                         f"Ry ({display_units['force']})", f"Mz ({display_units['moment']})"]
        react_data = [[], [], [], []]
        
        for node in sorted(self.nodes, key=lambda n: n.uid):
            if any(node.restraints) and hasattr(node, 'reactions'):
                if isinstance(node.reactions, dict) and combo in node.reactions:
                    reaction = node.reactions[combo]
                else:
                    reaction = node.reactions  # Direct array
                
                react_data[0].append(node.uid)
                
                # Only show reactions where there are restraints
                rx = extract_value(reaction[0]) if node.restraints[0] else 0
                ry = extract_value(reaction[1]) if node.restraints[1] else 0
                mz = extract_value(reaction[2]) if node.restraints[2] else 0
                
                react_data[1].append(f"{rx:.4f}")
                react_data[2].append(f"{ry:.4f}")
                react_data[3].append(f"{mz:.4f}")
        
        react_table = go.Figure(data=[go.Table(
            header=dict(
                values=react_headers,
                fill_color='lightblue',
                align='center',
                font=dict(size=12)
            ),
            cells=dict(
                values=react_data,
                fill_color='lavender',
                align='right',
                font=dict(size=11)
            )
        )])
        
        react_table.update_layout(
            title_text=f"Support Reactions - {combo}",
            height=400,
            margin=dict(l=0, r=0, b=0, t=30),
        )
        
        figures.append(react_table)
    
    # c. Member end forces table
    for combo in loadcombos:
        force_headers = ["Member", 
                         f"Axial i ({display_units['force']})", 
                         f"Shear i ({display_units['force']})", 
                         f"Moment i ({display_units['moment']})",
                         f"Axial j ({display_units['force']})", 
                         f"Shear j ({display_units['force']})", 
                         f"Moment j ({display_units['moment']})"]
        force_data = [[], [], [], [], [], [], []]
        
        for member in sorted(self.members, key=lambda m: m.uid):
            if hasattr(member, 'end_forces_local') and combo in member.end_forces_local:
                forces = member.end_forces_local[combo]
                
                force_data[0].append(member.uid)
                
                # Local end forces - force sign convention
                if len(forces) >= 6:
                    force_data[1].append(f"{extract_value(forces[0]):.4f}")  # Axial i
                    force_data[2].append(f"{extract_value(forces[1]):.4f}")  # Shear i
                    force_data[3].append(f"{extract_value(forces[2]):.4f}")  # Moment i
                    force_data[4].append(f"{extract_value(forces[3]):.4f}")  # Axial j
                    force_data[5].append(f"{extract_value(forces[4]):.4f}")  # Shear j
                    force_data[6].append(f"{extract_value(forces[5]):.4f}")  # Moment j
                else:
                    # Fill with zeros if not enough data
                    for i in range(1, 7):
                        force_data[i].append("0.0000")
        
        force_table = go.Figure(data=[go.Table(
            header=dict(
                values=force_headers,
                fill_color='lightblue',
                align='center',
                font=dict(size=12)
            ),
            cells=dict(
                values=force_data,
                fill_color='lavender',
                align='right',
                font=dict(size=11)
            )
        )])
        
        force_table.update_layout(
            title_text=f"Member End Forces (Local) - {combo}",
            height=400,
            margin=dict(l=0, r=0, b=0, t=30),
        )
        
        figures.append(force_table)
    
    # 6. Member properties table
    prop_headers = ["Member", 
                   f"Length ({display_units['length']})", 
                   "Section", 
                   "Material",
                   "Angle (deg)"]
    prop_data = [[], [], [], [], []]
    
    for member in sorted(self.members, key=lambda m: m.uid):
        prop_data[0].append(member.uid)
        prop_data[1].append(f"{extract_value(member.length):.4f}")
        
        # Get section name
        section_name = member.section.name if hasattr(member.section, 'name') else str(member.section.uid)
        prop_data[2].append(section_name)
        
        # Get material name
        material_name = str(member.material.uid)
        prop_data[3].append(material_name)
        
        # Orientation angle
        angle = np.degrees(member.theta) if hasattr(member, 'theta') else 0
        prop_data[4].append(f"{angle:.2f}")
    
    prop_table = go.Figure(data=[go.Table(
        header=dict(
            values=prop_headers,
            fill_color='lightblue',
            align='center',
            font=dict(size=12)
        ),
        cells=dict(
            values=prop_data,
            fill_color='lavender',
            align='right',
            font=dict(size=11)
        )
    )])
    
    prop_table.update_layout(
        title_text="Member Properties",
        height=400,
        margin=dict(l=0, r=0, b=0, t=30),
    )
    
    figures.append(prop_table)
    
    # Combine all figures into an HTML report with navigation tabs
    html_parts = ["""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Structural Analysis Results</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body { font-family: Arial, sans-serif; margin: 0; padding: 0; }
            .header { background-color: #2c3e50; color: white; padding: 20px; text-align: center; }
            .content { padding: 20px; }
            .tab { overflow: hidden; border: 1px solid #ccc; background-color: #f1f1f1; }
            .tab button { background-color: inherit; float: left; border: none; outline: none;
                          cursor: pointer; padding: 14px 16px; transition: 0.3s; }
            .tab button:hover { background-color: #ddd; }
            .tab button.active { background-color: #ccc; }
            .tabcontent { display: none; padding: 6px 12px; border: 1px solid #ccc; border-top: none; }
            .plot-container { width: 100%; height: 100%; }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Structural Analysis Results</h1>
            <p>Interactive visualization of analysis results</p>
        </div>
        
        <div class="tab">
    """]
    
    # Create tabs
    for i, fig in enumerate(figures):
        tab_title = fig.layout.title.text if hasattr(fig.layout, 'title') and hasattr(fig.layout.title, 'text') else f"Figure {i+1}"
        html_parts.append(f'<button class="tablinks" onclick="openTab(event, \'tab{i}\')">{tab_title}</button>')
    
    html_parts.append('</div>')
    
    # Create content for each tab
    for i, fig in enumerate(figures):
        html_parts.append(f'<div id="tab{i}" class="tabcontent">')
        html_parts.append(f'<div id="plot{i}" class="plot-container"></div>')
        html_parts.append('</div>')
        
    # Add JavaScript for tabs and plots
    html_parts.append("""
        <script>
            // Tab functionality
            function openTab(evt, tabName) {
                var i, tabcontent, tablinks;
                tabcontent = document.getElementsByClassName("tabcontent");
                for (i = 0; i < tabcontent.length; i++) {
                    tabcontent[i].style.display = "none";
                }
                tablinks = document.getElementsByClassName("tablinks");
                for (i = 0; i < tablinks.length; i++) {
                    tablinks[i].className = tablinks[i].className.replace(" active", "");
                }
                document.getElementById(tabName).style.display = "block";
                evt.currentTarget.className += " active";
            }
            
            // Open the first tab by default
            document.getElementsByClassName("tablinks")[0].click();
        </script>
    """)
    
    # Add plot data
    for i, fig in enumerate(figures):
        fig_json = fig.to_json()
        html_parts.append(f"""
        <script>
            var plotData{i} = {fig_json};
            Plotly.newPlot('plot{i}', plotData{i}.data, plotData{i}.layout);
        </script>
        """)
    
    html_parts.append("""
    </body>
    </html>
    """)
    
    # Write HTML file
    try:
        with open(output_html, 'w') as f:
            f.write(''.join(html_parts))
        print(f"Successfully created HTML report at: {output_html}")
        return output_html
    except Exception as e:
        print(f"Error writing HTML file: {e}")
        return None