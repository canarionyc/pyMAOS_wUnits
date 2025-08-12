def load_frame_from_file_new(filename, logger=None):
    """
    Reads a structural model from a JSON file, first converting it to SI units

    Parameters
    ----------
    filename : str
        Path to the input JSON file
    logger : logging.Logger, optional
        Logger object for output

    Returns
    -------
    tuple
        (node_list, element_list) ready for structural analysis, all in SI units
    """

    # Use print or pyMAOS.info based on what's available
    def pyMAOS.info(message):
        if logger:
            pyMAOS.info(message)
        else:
            print(message)

    import os
    import json
    import sys
    try:
        # Import the conversion utility
        sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), "examples"))
        from units.convert_units import convert_json_to_si

        # Create output filename with _SI suffix - without using Path module
        base_name = os.path.basename(filename)
        name_without_ext = os.path.splitext(base_name)[0]
        si_filename = os.path.join(os.path.dirname(filename), f"{name_without_ext}_SI.json")

        pyMAOS.info(f"Converting {filename} to SI units...")

        # Load the original JSON
        with open(filename, 'r') as file:
            data = json.load(file)

        # Convert to SI units and save
        convert_json_to_si(data, si_filename)
        pyMAOS.info(f"Converted data saved to {si_filename}")

        # Load the SI version using the original function
        pyMAOS.info(f"Loading SI converted model from {si_filename}...")
        return load_frame_from_file_new(str(si_filename), logger=logger)

    except ImportError:
        pyMAOS.info("Warning: Could not import convert_units.py. Falling back to standard loader.")
        return load_frame_from_file(filename, logger=logger)
    except Exception as e:
        pyMAOS.info(f"Error in unit conversion: {str(e)}")
        pyMAOS.info("Falling back to standard loader with unit conversions.")
        return load_frame_from_file(filename, logger=logger)