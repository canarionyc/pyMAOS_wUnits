from pyMAOS.model import Node, Element

def print_results(node_list, element_list, loadcombo):
    print("Displacements:")
    for i, node in enumerate(node_list):
        tx = node.displacements[loadcombo.name]
        print(f"N{node.uid} -- Ux: {tx[0]:.4E} -- Uy:{tx[1]:.4E} -- Rz:{tx[2]:.4E}")
    
    print("-" * 100)
    print("Reactions:")
    for i, node in enumerate(node_list):
        rx = node.reactions[loadcombo.name]
        print(f"N{node.uid} -- Rx: {rx[0]:.4E} -- Ry:{rx[1]:.4E} -- Mz:{rx[2]:.4E}")

    print("-" * 100)
    print("Member Forces:")
    for i, element in enumerate(element_list):
        fx = element.end_forces_local[loadcombo.name]

        print(f"M{element.uid}")
        print(
            f"    i -- Axial: {fx[0,0]:.4E} -- Shear: {fx[1,0]:.4E} -- Moment: {fx[2,0]:.4E}"
        )
        print(
            f"    j -- Axial: {fx[3,0]:.4E} -- Shear: {fx[4,0]:.4E} -- Moment: {fx[5,0]:.4E}"
        )
