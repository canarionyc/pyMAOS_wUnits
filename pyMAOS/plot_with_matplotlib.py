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
            dglobal = element.Dglobal_plot(loadcombo, disp_scale)
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

def plot_structure(
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
        dglobal = member.Dglobal_plot(loadcombo, displace_scale)

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


def plot_structure_matplotlib(
    nodes,
    members,
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
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot nodes and their labels
    for node in nodes:
        ax.plot(node.x, node.y, marker="o", markersize=6, color="red", label="Node" if "Node" not in ax.get_legend_handles_labels()[1] else "")
        ax.text(node.x, node.y + 5, f'N{node.uid}', color='darkred', fontsize=9, ha='center')

        # Visualize restraints
        if hasattr(node, "restraints"):
            rx, ry, rz = node.restraints
            size = 5  # Length of restraint symbol
            if rx:  # Horizontal restraint
                ax.plot([node.x - size, node.x + size], [node.y, node.y], color="blue", linewidth=2, label="Restraint (Ux)" if "Restraint (Ux)" not in ax.get_legend_handles_labels()[1] else "")
            if ry:  # Vertical restraint
                ax.plot([node.x, node.x], [node.y - size, node.y + size], color="green", linewidth=2, label="Restraint (Uy)" if "Restraint (Uy)" not in ax.get_legend_handles_labels()[1] else "")
            if rz:  # Rotational restraint
                theta = np.linspace(0, 2 * np.pi, 100)
                ax.plot(
                    node.x + (size / 2) * np.cos(theta),
                    node.y + (size / 2) * np.sin(theta),
                    color="purple",
                    linewidth=1.5,
                    label="Restraint (Rz)" if "Restraint (Rz)" not in ax.get_legend_handles_labels()[1] else "",
                )

    # Plot members and their labels
    for member in members:
        ax.plot(
            [member.inode.x, member.jnode.x],
            [member.inode.y, member.jnode.y],
            color="black",
            linewidth=1.5,
            label="Element" if "Element" not in ax.get_legend_handles_labels()[1] else "",
        )
        # Calculate midpoint for the label
        mid_x = (member.inode.x + member.jnode.x) / 2
        mid_y = (member.inode.y + member.jnode.y) / 2
        ax.text(mid_x, mid_y, f'M{member.uid}', color='darkblue', fontsize=9, ha='center', va='center', backgroundcolor=(1,1,1,0.7))


    # Add grid, labels, and legend
    ax.grid(True)
    ax.set_aspect("equal", "box")
    ax.set_title("Structure Visualization", fontsize=14)
    ax.set_xlabel("X Coordinate", fontsize=12)
    ax.set_ylabel("Y Coordinate", fontsize=12)
    ax.legend()

    # Return the figure and axis for further customization or saving
    return fig, ax
