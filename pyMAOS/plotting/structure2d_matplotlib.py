import numpy as np

def plot_element(element, loadcombo, scaling={
        "axial": 2,
        "displacement": 100,
    }):
    """
    Plots a single element with its nodes, geometry, and (if frame)
    internal force and deformed shape information.
    
    Parameters
    ----------
    element : R2Truss or R2Frame instance
        The structural element to plot.
    loadcombo : LoadCombo
        The load combination to evaluate deformations and forces.
    scaling : dict, optional
        Scaling factors for internal force and deformations.
    Returns
    -------
    fig, ax : matplotlib Figure and Axes objects
    """
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot the element's initial geometry and nodes.
    ax.plot(element.inode.x, element.inode.y, marker="o", markersize=6, color="red", label="In-Node")
    ax.plot(element.jnode.x, element.jnode.y, marker="s", markersize=6, color="green", label="J-Node")
    ax.plot([element.inode.x, element.jnode.x],
            [element.inode.y, element.jnode.y],
            color="black", linewidth=2, label="Element Geometry")
    
    ax.set_title(f"Element UID: {element.uid} ({element.type})", fontsize=14)
    
    # If the element is a frame, add deformed shape and axial force plot.
    if element.type != "TRUSS" and element._loaded:
        disp_scale = scaling.get("displacement", 100)
        axial_scale = scaling.get("axial", 2)
        
        try:
            # Get deformed shape (global displacements) and overlay it.c v
            dglobal = element.dglobal_span(loadcombo, disp_scale)
            ax.plot(dglobal[:, 0] + element.inode.x,
                    dglobal[:, 1] + element.inode.y,
                    linestyle="--", color="gray", label="Deformed Shape")
        except Exception as e:
            print(f"Error plotting deformed shape for element {element.uid}: {e}")
        
        try:
            # Plot axial force distribution (if available)
            aglobal = element.Aglobal_plot(loadcombo, axial_scale)
            ax.plot(aglobal[:, 0] + element.inode.x,
                    aglobal[:, 1] + element.inode.y,
                    linestyle=":", color="blue", label="Axial Force")
        except Exception as e:
            print(f"Error plotting axial force for element {element.uid}: {e}")
    
    ax.set_xlabel("X Coordinate")
    ax.set_ylabel("Y Coordinate")
    ax.grid(True)
    ax.legend()
    ax.set_aspect("equal", "box")
    fig.tight_layout()
    return fig, ax

def plot_structure_matplotlib_no_units(
    nodes,
    members,
    loadcombo,
    scaling={
        "axial_load": 100,
        "normal_load": 100,
        "point_load": 1,
        "axial": 2,
        "shear": 2,
        "moment": 0.1,
        "rotation": 5000,
        "displacement": 100,
    },
):
    # Plot the structure
    from matplotlib import pyplot as plt
    fig, axs = plt.subplots(2, 3, figsize=(12, 8)) # Increased figure size for clarity

    axial_loading_scale = scaling.get("axial_load", 1)
    normal_loading_scale = scaling.get("normal_load", 1)
    ptloading_scale = scaling.get("point_load", 1)
    axial_scale = scaling.get("axial", 1)
    shear_scale = scaling.get("shear", 1)
    moment_scale = scaling.get("moment", 1)
    rotation_scale = scaling.get("rotation", 1)
    displace_scale = scaling.get("displacement", 1)
    marker_shape = "o"
    marker_size = 2

    # --- Create a color map for different sections ---
    unique_sections = list(set(member.section for member in members))
    # Using a colormap to get distinct colors
    colors = plt.cm.get_cmap('viridis', len(unique_sections)) 
    section_color_map = {section: colors(i) for i, section in enumerate(unique_sections)}


    axs[0, 0].set_title(
        f"Geometry and Deformed Shape\n scale:{displace_scale}", fontsize=12
    )
    axs[0, 1].set_title(
        f"Member Loading \n axial scale:{axial_loading_scale} \n normal scale:{normal_loading_scale}\n pt load scale:{ptloading_scale}",
        fontsize=12,
    )

    axs[0, 2].set_title(f"Axial Force\n scale:{axial_scale}", fontsize=12)

    axs[1, 0].set_title(f"Shear Force\n scale:{shear_scale}", fontsize=12)

    axs[1, 1].set_title(f"Moment\n scale:{moment_scale}", fontsize=12)

    axs[1, 2].set_title(f"Cross-Section Rotation\n scale:{rotation_scale}", fontsize=12)

    for node in nodes:
        # Plot nodes across all subplots
        for ax in axs.flat:
            ax.plot(node.x, node.y, marker=".", markersize=8, color="red")
        
        # Plot deformed shape on the first subplot
        axs[0, 0].plot(
            node.x_displaced(loadcombo, displace_scale),
            node.y_displaced(loadcombo, displace_scale),
            marker=".",
            markersize=10,
            color="gray",
        )
        axs[0, 0].show()
        # --- Visualize restraints ---
        size = 6  # length of restraint symbol
        if hasattr(node, "restraints"):
            rx, ry, rz = node.restraints
            if rx:
                axs[0, 0].plot([node.x - size, node.x + size], [node.y, node.y], color="blue", linewidth=2)
            if ry:
                axs[0, 0].plot([node.x, node.x], [node.y - size, node.y + size], color="green", linewidth=2)
            if rz:
                theta = np.linspace(0, 2 * np.pi, 100)
                axs[0, 0].plot(node.x + (size/2) * np.cos(theta), node.y + (size/2) * np.sin(theta), color="purple", linewidth=1.5)

    for member in members: 
        member_color = section_color_map.get(member.section, "black") # Default to black if section not in map
        
        # Plot member geometry on all subplots, colored by section on the first one
        for i, ax in enumerate(axs.flat):
            color = member_color if i == 0 else "black" # Color by section only on the geometry plot
            ax.plot(
                [member.inode.x, member.jnode.x],
                [member.inode.y, member.jnode.y],
                linewidth=1.5,
                color=color,
            )

        aglobal = member.Aglobal_plot(loadcombo, axial_scale)
        dglobal = member.dglobal_span(loadcombo, displace_scale)

        axs[0, 2].plot(
            (aglobal[:, 0] + member.inode.x),
            (aglobal[:, 1] + member.inode.y),
            linewidth=1,
            color="blue",
            marker=marker_shape,
            markersize=marker_size,
        )

        axs[0, 0].plot(
            (dglobal[:, 0] + member.inode.x),
            (dglobal[:, 1] + member.inode.y),
            linewidth=1,
            color="gray",
            marker=marker_shape,
            markersize=marker_size,
        )

        if member.type != "TRUSS":
            vglobal = member.Vglobal_plot(loadcombo, shear_scale)
            mglobal = member.Mglobal_plot(loadcombo, moment_scale)
            sglobal = member.Sglobal_plot(loadcombo, rotation_scale)
            wxglobal = member.Wxglobal_plot(loadcombo, axial_loading_scale, ptloading_scale)
            wyglobal = member.Wyglobal_plot(loadcombo, normal_loading_scale, ptloading_scale)

            axs[0, 1].plot((wxglobal[:, 0] + member.inode.x), (wxglobal[:, 1] + member.inode.y), linewidth=1, color="blue", marker=marker_shape, markersize=marker_size)
            axs[0, 1].plot((wyglobal[:, 0] + member.inode.x), (wyglobal[:, 1] + member.inode.y), linewidth=1, color="red", marker=marker_shape, markersize=marker_size)
            axs[1, 0].plot((vglobal[:, 0] + member.inode.x), (vglobal[:, 1] + member.inode.y), linewidth=1, color="green", marker=marker_shape, markersize=marker_size)
            axs[1, 1].plot((mglobal[:, 0] + member.inode.x), (mglobal[:, 1] + member.inode.y), linewidth=1, color="red", marker=marker_shape, markersize=marker_size)
            axs[1, 1].plot((member.inode.x, mglobal[0, 0] + member.inode.x), (member.inode.y, mglobal[0, 1] + member.inode.y), linewidth=1, color="red")
            axs[1, 1].plot((member.jnode.x, mglobal[-1, 0] + member.inode.x), (member.jnode.y, mglobal[-1, 1] + member.inode.y), linewidth=1, color="red")
            axs[1, 2].plot((sglobal[:, 0] + member.inode.x), (sglobal[:, 1] + member.inode.y), linewidth=1, color="purple", marker=marker_shape, markersize=marker_size)

    for ax in axs.flat:
        ax.grid(True)
        ax.set_aspect("equal", "box")

    fig.tight_layout()
    return fig, axs


def plot_structure_matplotlib(nodes, members, ax=None, show_labels=True,
                             node_color='black', member_color='blue',
                             node_size=50, node_labels=True, member_labels=True):
    """
    Plot a 2D structure using matplotlib.

    Parameters
    ----------
    nodes : list
        A list of R2Node objects.
    members : list
        A list of R2Truss or R2Frame objects.
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, a new figure and axes are created.
    show_labels : bool, optional
        Whether to show node and member labels.
    node_color : str, optional
        Color for nodes.
    member_color : str, optional
        Color for members.
    node_size : int, optional
        Size of node markers.
    node_labels : bool, optional
        Whether to show node labels.
    member_labels : bool, optional
        Whether to show member labels.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    ax : matplotlib.axes.Axes
        The axes object.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Helper function to extract value from a potential Quantity object
    def extract_value(val):
        if hasattr(val, 'magnitude'):  # Check if it's a Pint Quantity
            return val.magnitude
        return val

    # Create figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 8))
    else:
        fig = ax.figure

    # Extract node coordinates, handling potential Quantity objects
    node_x = [extract_value(node.x) for node in nodes]; print(f"Node x: {node_x}")
    node_y = [extract_value(node.y) for node in nodes]; print(f"Node y: {node_y}")

    # Plot nodes
    ax.scatter(node_x, node_y, color=node_color, s=node_size, zorder=10)

    # Plot members
    for member in members:
        # Extract coordinates for both nodes, handling potential Quantity objects
        x1 = extract_value(member.inode.x)
        y1 = extract_value(member.inode.y)
        x2 = extract_value(member.jnode.x)
        y2 = extract_value(member.jnode.y)

        # Plot member as a line
        ax.plot([x1, x2], [y1, y2], color=member_color, linewidth=1.5)

        # Add member label at midpoint if requested
        if show_labels and member_labels:
            mid_x = (x1 + x2) / 2
            mid_y = (y1 + y2) / 2
            ax.text(mid_x, mid_y, f"M{member.uid}", color=member_color,
                   ha='center', va='center', fontsize=8,
                   bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.1'))

    # Add node labels if requested
    if show_labels and node_labels:
        for i, node in enumerate(nodes):
            ax.text(node_x[i], node_y[i], f"N{node.uid}", color=node_color,
                   ha='left', va='bottom', fontsize=8, fontweight='bold')

    # Add hinges for frame members with hinges
    for member in members:
        if hasattr(member, 'hinges') and any(member.hinges):
            x1 = extract_value(member.inode.x)
            y1 = extract_value(member.inode.y)
            x2 = extract_value(member.jnode.x)
            y2 = extract_value(member.jnode.y)

            if member.hinges[0]:  # i-node hinge
                ax.scatter(x1, y1, marker='o', facecolors='none', edgecolors='red', s=80, zorder=11)

            if member.hinges[1]:  # j-node hinge
                ax.scatter(x2, y2, marker='o', facecolors='none', edgecolors='red', s=80, zorder=11)

    # Adjust plot appearance
    ax.set_aspect('equal')
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Structure Plot')

    # Add some padding around the structure
    x_min, x_max = min(node_x), max(node_x)
    y_min, y_max = min(node_y), max(node_y)
    padding = max(x_max - x_min, y_max - y_min) * 0.1
    ax.set_xlim(x_min - padding, x_max + padding)
    ax.set_ylim(y_min - padding, y_max + padding)

    # Make sure the figure is tight
    fig.tight_layout()

    return fig, ax
