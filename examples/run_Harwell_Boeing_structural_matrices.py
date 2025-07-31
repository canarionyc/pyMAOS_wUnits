import numpy as np
import scipy.io as sio
import os
import requests
import matplotlib.pyplot as plt
from pyMAOS.structure2d import R2Structure

def download_matrix(matrix_id, target_dir="."):
    """Download a matrix from Matrix Market."""
    base_url = "https://math.nist.gov/pub/MatrixMarket2/Harwell-Boeing/bcsstruc1"
    filename = f"{matrix_id}.mtx.gz"
    url = f"{base_url}/{filename}"

    local_path = os.path.join(target_dir, filename)
    if not os.path.exists(local_path):
        print(f"Downloading {url}...")
        r = requests.get(url)
        with open(local_path, 'wb') as f:
            f.write(r.content)
        print(f"Saved to {local_path}")
    return local_path

def load_hb_matrix(matrix_id, target_dir="."):
    """Load a Harwell-Boeing matrix into memory."""
    filepath = download_matrix(matrix_id, target_dir)

    # Use scipy to read the matrix
    print(f"Loading matrix {matrix_id}...")
    matrix = sio.mmread(filepath)
    return matrix.tocsr()  # Convert to CSR format for efficiency

def visualize_sparsity(matrix, title="Sparsity Pattern"):
    """Visualize the sparsity pattern of a matrix."""
    plt.figure(figsize=(10, 10))
    plt.spy(matrix, markersize=0.5)
    plt.title(f"{title} - Shape: {matrix.shape}")
    plt.show()

def create_structure_from_matrix(K, force_vector=None):
    """
    Convert a stiffness matrix to an R2Structure object.

    This is a simplified approach - in practice you would need to:
    1. Interpret DOFs correctly
    2. Extract node and element information
    3. Map loads correctly
    """
    # Create a minimal structure with the correct DOF count
    n_dof = K.shape[0]
    n_nodes = n_dof // 3  # Assuming 3 DOF per node for 2D structures

    # Create dummy nodes and elements
    from pyMAOS.node2d import R2Node
    from pyMAOS.frame2d import R2Frame
    from pyMAOS.material import LinearElasticMaterial
    from pyMAOS.section import Section

    nodes = [R2Node(i, x=i, y=0) for i in range(n_nodes)]

    # Create a minimal structure
    structure = R2Structure(nodes, [])

    # Directly set the stiffness matrix
    structure.KSTRUCT = K.toarray()

    # If force vector is provided, set it
    if force_vector is not None:
        structure.FM = force_vector

    return structure

# Example usage
if __name__ == "__main__":
    # Download and load a specific matrix (e.g., "bcsstk01")
    matrix_id = "bcsstk01"
    K = load_hb_matrix(matrix_id)

    print(f"Matrix shape: {K.shape}")
    print(f"Number of non-zeros: {K.nnz}")

    # Visualize the sparsity pattern
    visualize_sparsity(K, f"Stiffness Matrix {matrix_id}")

    # Create a force vector (all zeros except for a few loads)
    F = np.zeros(K.shape[0])
    F[K.shape[0]//2] = 1000  # Apply a load at the middle DOF

    # Create a structure from the matrix
    structure = create_structure_from_matrix(K, F)

    # Now you can use your existing solver
    from scipy.sparse.linalg import spsolve
    u = spsolve(K, F)
    print(f"Max displacement: {np.max(np.abs(u))}")

    # You could then use your R2Structure methods for post-processing
    # structure.U = u  # Set the displacement vector
    # structure.further_analysis(...)  # Process results