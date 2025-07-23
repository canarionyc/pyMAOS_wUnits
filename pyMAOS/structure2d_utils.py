def __str__(self):
    """
    Returns a string representation of the structure using the display units
    specified in the input file.
    """
    # Use units from self.units dictionary with appropriate defaults
    force_unit = self.units.get('force', 'N')
    length_unit = self.units.get('length', 'm')
    pressure_unit = self.units.get('pressure', 'Pa')
    moment_unit = self.units.get('moment', f"{force_unit}*{length_unit}")
    distributed_load_unit = self.units.get('distributed_load', f"{force_unit}/{length_unit}")
    
    # Create a ureg for conversions if not already available
    try:
        from pint import UnitRegistry
        ureg = UnitRegistry()
        Q_ = ureg.Quantity
    except:
        # Fall back if pint is not available
        print("Warning: pint library not available, using SI units for display")
        Q_ = lambda x, unit: x

    # Build the output string
    result = []
    result.append("=" * 80)
    result.append(f"STRUCTURAL MODEL SUMMARY")
    result.append(f"Display units: Force={force_unit}, Length={length_unit}, Pressure={pressure_unit}")
    result.append("-" * 80)
    
    # Basic structure info
    result.append(f"Number of nodes: {self.NJ}")
    result.append(f"Number of members: {self.NM}")
    result.append(f"Degrees of freedom: {self.NDOF}")
    result.append(f"Number of restraints: {self.NR}")
    
    # Node information
    result.append("\nNODE INFORMATION:")
    result.append(f"{'Node ID':8} {'X ('+length_unit+')':15} {'Y ('+length_unit+')':15} {'Restraints':15}")
    result.append("-" * 80)
    
    import operator
    for node in sorted(self.nodes, key=operator.attrgetter('uid')):
        # Convert coordinates to display units
        x_display = Q_(node.x, 'm').to(length_unit).magnitude if hasattr(Q_, 'to') else node.x
        y_display = Q_(node.y, 'm').to(length_unit).magnitude if hasattr(Q_, 'to') else node.y
        
        # Format restraints
        restraint_str = "".join([
            "Rx" if node.restraints[0] else "--",
            " Ry" if node.restraints[1] else " --",
            " Rz" if node.restraints[2] else " --"
        ])
        
        result.append(f"{node.uid:<8} {x_display:<15.4g} {y_display:<15.4g} {restraint_str:<15}")

    # Member information
    result.append("\nMEMBER INFORMATION:")
    result.append(f"{'Member ID':10} {'Type':8} {'i-node':8} {'j-node':8} {'Length ('+length_unit+')':15}")
    result.append("-" * 80)
    
    for member in sorted(self.members, key=lambda m: m.uid): # type: ignore
        # Convert length to display units
        length_display = Q_(member.length, 'm').to(length_unit).magnitude if hasattr(Q_, 'to') else member.length
        
        result.append(f"{member.uid:<10} {member.type:<8} {member.inode.uid:<8} {member.jnode.uid:<8} {length_display:<15.4g}")

    # Material properties
    materials_seen = set()
    result.append("\nMATERIAL PROPERTIES:")
    result.append(f"{'Material':10} {'E ('+pressure_unit+')':15}")
    result.append("-" * 80)
    
    for member in self.members:
        if member.material not in materials_seen:
            # Convert elastic modulus to display units
            E_display = Q_(member.material.E, 'Pa').to(pressure_unit).magnitude if hasattr(Q_, 'to') else member.material.E
            
            result.append(f"{member.material.uid:<10} {E_display:<15.4g}")
            materials_seen.add(member.material)

    # Section properties
    sections_seen = set()
    result.append("\nSECTION PROPERTIES:")
    result.append(f"{'Section':10} {'Area ('+length_unit+'^2)':18} {'Ixx ('+length_unit+'^4)':18}")
    result.append("-" * 80)
    
    for member in self.members:
        if member.section not in sections_seen:
            # Convert section properties to display units
            area_display = Q_(member.section.Area, 'm^2').to(length_unit+'^2').magnitude if hasattr(Q_, 'to') else member.section.Area
            ixx_display = Q_(member.section.Ixx, 'm^4').to(length_unit+'^4').magnitude if hasattr(Q_, 'to') else member.section.Ixx
            
            result.append(f"{member.section.uid:<10} {area_display:<18.4g} {ixx_display:<18.4g}")
            sections_seen.add(member.section)

    # Load information
    has_node_loads = any(node.loads for node in self.nodes)
    if has_node_loads:
        result.append("\nNODAL LOADS:")
        result.append(f"{'Node ID':8} {'Load Case':10} {'Fx ('+force_unit+')':15} {'Fy ('+force_unit+')':15} {'Mz ('+moment_unit+')':15}")
        result.append("-" * 80)
        
        for node in sorted(self.nodes, key=lambda n: n.uid): # type: ignore
            if node.loads:
                for case, load in node.loads.items():
                    # Convert forces and moments to display units
                    fx_display = Q_(load[0], 'N').to(force_unit).magnitude if hasattr(Q_, 'to') else load[0]
                    fy_display = Q_(load[1], 'N').to(force_unit).magnitude if hasattr(Q_, 'to') else load[1]
                    mz_display = Q_(load[2], 'N*m').to(moment_unit).magnitude if hasattr(Q_, 'to') else load[2]
                    
                    result.append(f"{node.uid:<8} {case:<10} {fx_display:<15.4g} {fy_display:<15.4g} {mz_display:<15.4g}")

    # Displacements and reactions if calculated
    has_displacements = any(hasattr(node, 'displacements') and node.displacements for node in self.nodes)
    if has_displacements:
        result.append("\nNODAL DISPLACEMENTS (most recent load combination):")
        result.append(f"{'Node ID':8} {'Ux ('+length_unit+')':15} {'Uy ('+length_unit+')':15} {'Rz (rad)':15}")
        result.append("-" * 80)
        
        for node in sorted(self.nodes, key=lambda n: n.uid): # type: ignore
            if hasattr(node, 'displacements') and node.displacements:
                # Get the most recent load combination
                latest_combo = list(node.displacements.keys())[-1]
                disp = node.displacements[latest_combo]
                
                # Convert displacements to display units
                ux_display = Q_(disp[0], 'm').to(length_unit).magnitude if hasattr(Q_, 'to') else disp[0]
                uy_display = Q_(disp[1], 'm').to(length_unit).magnitude if hasattr(Q_, 'to') else disp[1]
                rz_display = disp[2]  # Radians are dimensionless
                
                result.append(f"{node.uid:<8} {ux_display:<15.6g} {uy_display:<15.6g} {rz_display:<15.6g}")
    
    # Errors if any
    if self._ERRORS:
        result.append("\nERRORS:")
        result.append("-" * 80)
        for error in self._ERRORS:
            result.append(error)
    
    # Return the joined result
    return "\n".join(result)
