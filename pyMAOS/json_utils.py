import json
import jsonschema
from jsonschema import validators
def extend_with_dimension_validator(validator_class):
    """Extend JSON Schema validator with dimension validation"""
    validate_properties = validator_class.VALIDATORS["properties"]

    def set_dimensions(validator, properties, instance, schema):
        for property, subschema in properties.items():
            if instance is not None and property in instance:
                # Process normal validation
                for error in validate_properties(validator, properties, instance, schema):
                    yield error

                # Add dimension validation if specified
                if "dimension" in subschema and instance[property] is not None:
                    dimension = subschema["dimension"]
                    value = instance[property]

                    # Handle string values with units
                    if isinstance(value, str):
                        try:
                            # Parse and validate the unit dimensions
                            parsed = parse_value_with_units(value)
                            if isinstance(parsed, pint.Quantity):
                                # Get the expected dimension's internal unit
                                expected_unit = get_internal_unit(dimension)
                                if expected_unit:
                                    try:
                                        # Try converting - will fail if dimensions don't match
                                        parsed.to(expected_unit)
                                    except pint.DimensionalityError as e:
                                        yield jsonschema.ValidationError(
                                            f"Value '{value}' has incorrect dimension for '{property}'. "
                                            f"Expected {dimension} ({expected_unit}), got {parsed.units}")
                        except Exception as e:
                            yield jsonschema.ValidationError(
                                f"Failed to parse unit dimensions for '{property}': {str(e)}")

    return validators.extend(validator_class, {"properties": set_dimensions})


# Create our custom validator
DimensionValidator = extend_with_dimension_validator(jsonschema.Draft7Validator)


def validate_input_with_schema(input_file, schema_file=None):
    """
    Validate a JSON input file against the schema with dimension validation

    Parameters
    ----------
    input_file : str
        Path to the input JSON file to validate
    schema_file : str, optional
        Path to the schema file (defaults to data/myschema.json)

    Returns
    -------
    bool
        True if validation succeeds

    Raises
    ------
    ValidationError
        If validation fails
    """
    # Default schema location
    # if schema_file is None:
    #     # Try to find schema in a few common locations
    #     possible_paths = [
    #         Path("data/myschema.json"),
    #         Path("myschema.json"),
    #         Path(__file__).parent.parent / "data" / "myschema.json"
    #     ]
    #
    #     for path in possible_paths:
    #         if path.exists():
    #             schema_file = str(path)
    #             break
    #     else:
    #         raise FileNotFoundError("Could not find schema file 'myschema.json'")

    # Load the schema
    with open(schema_file, 'r') as f:
        schema = json.load(f)

    # Load the input file
    with open(input_file, 'r') as f:
        data = json.load(f)

    # Create validator instance
    validator = DimensionValidator(schema)

    # Collect and report all errors
    errors = list(validator.iter_errors(data))
    if errors:
        print(f"Validation failed with {len(errors)} errors:")
        for i, error in enumerate(errors, 1):
            # Format the error path as a JSON path
            path = ".".join(str(p) for p in error.path) if error.path else "root"
            print(f"{i}. Error at '{path}': {error.message}")

        # Raise the first error to stop execution if needed
        raise jsonschema.ValidationError(f"Validation failed: {errors[0].message}")

    print("Input file successfully validated against schema!")
    return True
