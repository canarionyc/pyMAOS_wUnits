def is_class_imported(class_name):
    """
    Check if a class has been imported in the current namespace

    Parameters
    ----------
    class_name : str
        Name of the class to check

    Returns
    -------
    bool
        True if class exists, False otherwise
    """
    return class_name in globals() or class_name in locals()


def check_class_exists(class_name):
    """Check if a class is available in the current namespace"""
    try:
        # Try to evaluate the class name
        return eval(class_name) is not None
    except (NameError, AttributeError):
        return False


def is_class_available_from_module(module_name, class_name):
    """Check if a class is available from a specific imported module"""
    import sys

    # Check if module is imported
    if module_name not in sys.modules:
        return False

    # Get the module object
    module = sys.modules[module_name]

    # Check if class exists in module
    return hasattr(module, class_name)


def list_imported_classes(module_filter=None):
    """
    List all classes that have been imported in the current Python script

    Parameters
    ----------
    module_filter : str or list of str, optional
        Filter classes by module name prefix (e.g., 'pyMAOS' or ['pyMAOS', 'numpy'])

    Returns
    -------
    dict
        Dictionary mapping class names to their module names
    """
    import inspect
    import sys

    # Normalize filter to a list
    if module_filter is None:
        filters = None
    elif isinstance(module_filter, str):
        filters = [module_filter]
    else:
        filters = list(module_filter)

    classes_dict = {}

    # Check global namespace
    for name, obj in globals().items():
        if inspect.isclass(obj):
            module = inspect.getmodule(obj)
            if module:
                module_name = module.__name__
                if filters is None or any(module_name.startswith(m) for m in filters):
                    classes_dict[name] = module_name

    # Check modules in sys.modules
    for module_name, module in sys.modules.items():
        # Skip None modules or if filtering is active and module doesn't match
        if module is None:
            continue
        if filters and not any(module_name.startswith(m) for m in filters):
            continue

        try:
            for name, obj in inspect.getmembers(module, inspect.isclass):
                # Only include if defined in this module (not imported into it)
                if hasattr(obj, '__module__') and obj.__module__ == module_name:
                    classes_dict[name] = module_name
        except:
            # Some modules might raise errors when inspected
            raise Warning(
                f"Could not inspect module {module_name}. It may not be a valid Python module or may not support introspection.")
            pass

    return classes_dict


# Example usage:
def print_imported_classes(module_filter=None):
    """Print imported classes in a formatted table"""
    classes = list_imported_classes(module_filter)

    print(f"\n{'Class Name':<30} | {'Module'}")
    print("-" * 60)

    for name, module in sorted(classes.items()):
        print(f"{name:<30} | {module}")

    print(f"\nTotal: {len(classes)} classes found")