import numpy as np

def numpy_array_renderer(ndarray_obj, precision=3):
	"""
	Custom renderer for numpy arrays in PyCharm debugger.
	Displays arrays with limited decimal precision.

	Args:
		ndarray_obj: The numpy array to render
		precision: Number of decimal places to show (default: 3)

	Returns:
		String representation of the array with controlled precision
	"""
	# Save current numpy print settings
	old_precision = np.get_printoptions()['precision']
	old_suppress = np.get_printoptions()['suppress']

	# Set temporary print settings
	np.set_printoptions(precision=precision, suppress=True)

	# Generate string representation
	result = str(ndarray_obj)

	# Restore original print settings
	np.set_printoptions(precision=old_precision, suppress=old_suppress)

	return result


# For direct use in PyCharm debugger
def format_array(array_obj):
	"""
	Direct formatter function for PyCharm debugger
	"""
	return numpy_array_renderer(array_obj)


# Fixed renderers with specific precision
def format_array_1d(array_obj):
	"""Format with 1 decimal place"""
	return numpy_array_renderer(array_obj, precision=1)

def format_array_2d(array_obj):
	"""Format with 2 decimal places"""
	return numpy_array_renderer(array_obj, precision=2)

def format_array_3d(array_obj):
	"""Format with 3 decimal places"""
	return numpy_array_renderer(array_obj, precision=3)


# Direct inline renderer for PyCharm - no imports needed
def render_ndarray(arr):
	"""
	Direct renderer that can be copy-pasted into PyCharm
	"""
	import numpy as np
	old_opts = np.get_printoptions()
	np.set_printoptions(precision=3, suppress=True)
	result = str(arr)
	np.set_printoptions(**old_opts)
	return result

# Working inline renderers for PyCharm that actually format the output
def get_pycharm_renderer_expression(precision=3):
	"""
	Generate a working PyCharm debugger expression for numpy arrays
	"""
	# This expression works in PyCharm's debugger without imports
	return f"'\\n'.join([' '.join([f'{{:.{precision}f}}'.format(val) for val in row]) if self.ndim > 1 else f'{{:.{precision}f}}'.format(row) for row in (self if self.ndim > 1 else [self])])"

# Pre-made expressions for common precisions
PYCHARM_RENDERER_1D = "'\\n'.join([f'{val:.1f}' for val in self])"
PYCHARM_RENDERER_2D = "'\\n'.join([' '.join([f'{val:.2f}' for val in row]) for row in self])"
PYCHARM_RENDERER_3D = "'\\n'.join([' '.join([f'{val:.3f}' for val in row]) for row in self])"

# Universal renderer that handles any ndarray shape
PYCHARM_RENDERER_UNIVERSAL = "repr(self.__class__(self.round(3)))[6:-1]"  # Strips 'array(' and ')'
