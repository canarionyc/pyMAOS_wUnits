def save_structure_state(self, filename):
    """
    Save the current state of the structure to a binary file for later restart.
    """
    import pickle
    import datetime
    import zlib

    print(f"Saving structure state to {filename}...")

    # Create a dictionary with all the state we want to save
    state = {
        # Core structure components
        'nodes': self.nodes,
        'members': self.members,
        'NJ': self.NJ,
        'NM': self.NM,
        'NR': self.NR,
        'DIM': self.DIM,
        'NDOF': self.NDOF,

        # Support data
        '_springNodes': self._springNodes,
        '_nonlinearNodes': self._nonlinearNodes,

        # Solution data
        '_D': self._D,
        'U': self.U,
        'FM': getattr(self, 'FM', None),
        'KSTRUCT': getattr(self, 'KSTRUCT', None),
        'USTRUCT': getattr(self, 'USTRUCT', None),
        'FG': getattr(self, 'FG', None),
        'structure_fef': getattr(self, 'structure_fef', None),
        'Kff': getattr(self, 'Kff', None),
        'FGf': getattr(self, 'FGf', None),
        'PFf': getattr(self, 'PFf', None),

        # Flags and errors
        '_unstable': self._unstable,
        '_Kgenerated': self._Kgenerated,
        '_ERRORS': self._ERRORS,

        # Units
        'units': self.units if hasattr(self, 'units') else None,

        # Adding plot_enabled flag
        'plot_enabled': False,

        # Metadata
        'save_timestamp': datetime.datetime.now().isoformat(),
    }

    # Debug information
    print(f"State dictionary contains keys: {list(state.keys())}")
    print(f"Nodes count: {len(state['nodes'])}")
    print(f"Members count: {len(state['members'])}")

    try:
        # Pickle and compress the data
        pickled_data = pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL)
        compressed_data = zlib.compress(pickled_data)

        with open(filename, 'wb') as f:
            f.write(compressed_data)

        print(f"Structure state saved successfully to {filename}")
        print(f"File size: {len(compressed_data) / 1024:.2f} KB")
        return True
    except Exception as e:
        print(f"Error saving structure state: {e}")
        import traceback
        traceback.print_exc()
        return False


def load_structure_state(self, filename):
    """
    Load structure state from a binary file

    Parameters
    ----------
    filename : str
        Path to the binary file

    Returns
    -------
    bool
        True if loading was successful, False otherwise
    """
    import pickle
    import zlib
    import pint
    import io

    print(f"Loading structure state from {filename}...")

    try:
        # Read the compressed data
        with open(filename, 'rb') as file:
            compressed_data = file.read()

        # Monkey patch the unit_manager.ureg function to handle Quantity objects correctly
        original_ureg = pyMAOS.unit_manager.ureg

        def patched_ureg(value, *args, **kwargs):
            if isinstance(value, pint.Quantity):
                # If already a Quantity, convert to current registry
                return pyMAOS.unit_manager.ureg.Quantity(value.magnitude, str(value.units))
            return original_ureg(value, *args, **kwargs)

        # Apply the patch before unpickling
        pyMAOS.unit_manager.ureg = patched_ureg

        try:
            # Try to decompress and unpickle the data
            try:
                pickled_data = zlib.decompress(compressed_data)

                # Define a custom unpickler for Pint Quantity objects
                class QuantityUnpickler(pickle.Unpickler):
                    def find_class(self, module, name):
                        if module == 'pint.quantity' and name == 'Quantity':
                            return pyMAOS.unit_manager.ureg.Quantity
                        return super().find_class(module, name)

                # Use custom unpickler
                unpickler = QuantityUnpickler(io.BytesIO(pickled_data))
                state_data = unpickler.load();
                from pprint import pprint;
                pprint(state_data)
                print(state_data.keys())
                print("Successfully loaded compressed state data")

            except zlib.error:
                # If decompression fails, try as raw pickle
                print("Data doesn't appear to be compressed, trying direct pickle")
                with open(filename, 'rb') as file:
                    unpickler = QuantityUnpickler(file)
                    state_data = unpickler.load()

            # Process the loaded data
            for key, value in state_data.items():
                if hasattr(self, key):
                    setattr(self, key, value)

            # Rebuild node and member mappings
            self.create_uid_maps()

            # After successful load, verify the state
            if self.verify_structure_state(state_data):
                print("✓ Structure state verified successfully")
                return True
            else:
                print("⚠ Structure state verification failed")
                return False

        finally:
            # Always restore the original ureg function
            pyMAOS.unit_manager.ureg = original_ureg

    except Exception as e:
        print(f"Error loading structure state: {e}")
        import traceback
        traceback.print_exc()
        return False