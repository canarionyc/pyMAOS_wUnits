def save_structure_state(self, filename):
    """
    Save the current state of the structure to a binary file for later restart.

    This function saves all necessary matrices, vectors, and structure data
    to allow restarting the program without recalculation.

    Parameters
    ----------
    filename : str
        Filename where the state will be saved

    Returns
    -------
    bool
        True if successful, False otherwise
    """
    import pickle
    import datetime
    import zlib

    print(f"Saving structure state to {filename}...")

    # Create a dictionary with all the state we want to save
    state = {
        # Matrices and vectors
        'KSTRUCT': getattr(self, 'KSTRUCT', None),
        'FM': getattr(self, 'FM', None),
        'Kff': getattr(self, 'Kff', None),
        'FGf': getattr(self, 'FGf', None),
        'PFf': getattr(self, 'PFf', None),
        'U': getattr(self, 'U', None),

        # Structure dimensions
        'NDOF': getattr(self, 'NDOF', None),
        'NNR': getattr(self, 'NNR', None),

        # Structure components
        'nodes': getattr(self, 'nodes', None),
        'elements': getattr(self, 'elements', None),
        'loads': getattr(self, 'loads', None),
        'supports': getattr(self, 'supports', None),

        # Metadata
        'timestamp': datetime.datetime.now().isoformat(),
        'description': f"Structure state saved at {datetime.datetime.now()}"
    }

    try:
        # Pickle and compress the data
        pickled_data = pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)
        compressed_data = zlib.compress(pickled_data)

        with open(filename, 'wb') as f:
            f.write(compressed_data)

        print(f"Structure state saved successfully to {filename}")
        print(f"File size: {len(compressed_data)/1024:.2f} KB")
        return True
    except Exception as e:
        print(f"Error saving structure state: {e}")
        return False

def load_structure_state(self, filename):
    """
    Load a previously saved structure state from a binary file.

    This function restores all matrices, vectors, and structure data
    to continue analysis without recalculation.

    Parameters
    ----------
    filename : str
        Filename containing the saved state

    Returns
    -------
    bool
        True if successful, False otherwise
    """
    import pickle
    import zlib

    print(f"Loading structure state from {filename}...")

    try:
        with open(filename, 'rb') as f:
            compressed_data = f.read()

        # Decompress and unpickle
        decompressed_data = zlib.decompress(compressed_data)
        state = pickle.loads(decompressed_data)

        # Restore all state variables
        for key, value in state.items():
            if key != 'timestamp' and key != 'description' and value is not None:
                setattr(self, key, value)
                if isinstance(value, dict) or hasattr(value, '__len__'):
                    print(f"Restored {key} with {len(value)} items")
                else:
                    print(f"Restored {key}")

        print(f"Structure state loaded successfully from {filename}")
        print(f"Original save timestamp: {state.get('timestamp', 'unknown')}")

        return True
    except Exception as e:
        print(f"Error loading structure state: {e}")
        import traceback
        traceback.print_exc()
        return False