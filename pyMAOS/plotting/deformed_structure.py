def plot_deformed_structure(nodes, members, loads, scaling=None, figsize=(12, 8)):
    """
    Plot the structure with deformations using PiecewisePolynomial2 displacement data.

    Parameters
    ----------
    nodes : list
        List of structural nodes
    members : list
        List of structural members
    loads : list
        List of load objects (LinearLoadXY) with Dy2 attributes
    scaling : dict, optional
        Dictionary with scaling factors for different response types
    figsize : tuple, optional
        Figure size (width, height) in inches

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
        The figure and axes containing the plot
    """
    import matplotlib.pyplot as plt
    import numpy as np

    # Default scaling if not provided
    if scaling is None:
        scaling = {
            "displacement": 100,
        }

    displace_scale = scaling.get("displacement", 100)

    # Create figure and axes
    fig, ax = plt.subplots(figsize=figsize)

    # Create a mapping from member UID to its loads
    member_load_map = {}
    for load in loads:
        if hasattr(load, 'member_uid'):
            member_load_map[load.member_uid] = load

    # Plot the original structure
    for member in members:
        # Extract coordinates (handle Quantity objects if present)
        x1 = member.inode.x.magnitude if hasattr(member.inode.x, 'magnitude') else member.inode.x
        y1 = member.inode.y.magnitude if hasattr(member.inode.y, 'magnitude') else member.inode.y
        x2 = member.jnode.x.magnitude if hasattr(member.jnode.x, 'magnitude') else member.jnode.x
        y2 = member.jnode.y.magnitude if hasattr(member.jnode.y, 'magnitude') else member.jnode.y

        # Plot original member
        ax.plot([x1, x2], [y1, y2], 'k-', linewidth=1.5, alpha=0.6, label='_Original')

        # Plot nodes
        ax.plot(x1, y1, 'ro', markersize=6, label='_Node')
        ax.plot(x2, y2, 'ro', markersize=6)

        # Check if this member has load data with displacement polynomial
        if member.uid in member_load_map:
            load = member_load_map[member.uid]

            if hasattr(load, 'Dy2') and load.Dy2.ppoly is not None:
                print(f"Processing deformation for member {member.uid}")

                # Calculate length and direction cosines
                dx = x2 - x1
                dy = y2 - y1
                length = np.sqrt(dx**2 + dy**2)
                cos_theta = dx / length
                sin_theta = dy / length

                # Generate points along the member
                num_points = 50
                x_local = np.linspace(0, length, num_points)

                try:
                    # Get displacement values
                    y_local = load.Dy2.evaluate_vectorized(x_local)

                    # Extract magnitude if it's a Quantity object
                    if hasattr(y_local, 'magnitude'):
                        y_local = y_local.magnitude * displace_scale
                    else:
                        y_local = y_local * displace_scale

                    # Transform to global coordinates
                    x_global = []
                    y_global = []
                    for i in range(len(x_local)):
                        # Local to global transformation
                        xg = x1 + x_local[i] * cos_theta - y_local[i] * sin_theta
                        yg = y1 + x_local[i] * sin_theta + y_local[i] * cos_theta
                        x_global.append(xg)
                        y_global.append(yg)

                    # Plot deformed shape
                    ax.plot(x_global, y_global, 'b-', linewidth=2, label='_Deformed')

                    print(f"Max displacement for member {member.uid}: {np.max(np.abs(y_local)) / displace_scale:.5e}")

                except Exception as e:
                    print(f"Error plotting deformation for member {member.uid}: {e}")

    # Set up plot appearance
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_aspect('equal')
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_title(f'Structural Deformation (Scale: {displace_scale}×)')

    # Create legend without duplicate entries
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    filtered_labels = [l for l in by_label.keys() if not l.startswith('_')]
    if filtered_labels:
        ax.legend([by_label[l] for l in filtered_labels], filtered_labels)

    fig.tight_layout()
    return fig, ax