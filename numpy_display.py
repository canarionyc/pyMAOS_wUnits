import numpy as np
import scipy as sp

def format_array(arr, precision=3, max_elements=50, with_metadata=True):
	"""
	Format a numpy array with controlled precision for better display.

	Args:
		arr: NumPy array to format
		precision: Number of decimal places to show (default: 3)
		max_elements: Maximum number of elements to display (default: 50)
		with_metadata: Whether to include shape and dtype info (default: True)

	Returns:
		Formatted string representation of the array
	"""
	# Save original print options
	old_precision = np.get_printoptions()['precision']
	old_suppress = np.get_printoptions()['suppress']
	old_threshold = np.get_printoptions()['threshold']

	# Set custom print options
	np.set_printoptions(precision=precision, suppress=True, threshold=max_elements)

	# Get the string representation
	result = str(arr)

	# Add metadata if requested
	if with_metadata:
		shape_str = 'x'.join(map(str, arr.shape))
		dtype_str = str(arr.dtype)
		result += f" (shape: {shape_str}, dtype: {dtype_str})"

	# Restore original print options
	np.set_printoptions(precision=old_precision, suppress=old_suppress, threshold=old_threshold)

	return result

def set_custom_repr(precision=3, threshold=50, max_width=100):
	"""
	Set a custom __repr__ method for NumPy arrays globally.
	This affects all array displays in the current session.

	Args:
		precision: Number of decimal places to show (default: 3)
		threshold: Maximum number of array elements to show (default: 50)
		max_width: Maximum line width for array display (default: 100)
	"""
	# Store original repr if not already stored
	if not hasattr(np.ndarray, '_original_repr'):
		np.ndarray._original_repr = np.ndarray.__repr__

	def custom_repr(self):
		# Save original print options
		old_opts = np.get_printoptions()

		# Set custom print options
		np.set_printoptions(precision=precision, suppress=True,
						   threshold=threshold, linewidth=max_width)

		# Get the string representation
		result = np.array2string(self, precision=precision, suppress_small=True,
								separator=' ', prefix='')

		# Restore original print options
		np.set_printoptions(**old_opts)

		# Format the final output
		class_name = self.__class__.__name__
		shape_str = 'x'.join(map(str, self.shape))
		dtype_str = str(self.dtype)

		return f"{class_name}({result}, shape=({shape_str}), dtype={dtype_str})"

	# Apply the custom __repr__ method
	np.ndarray.__repr__ = custom_repr

	print(f"NumPy array display customized with {precision} decimal places, {threshold} max elements")

	return custom_repr  # Return for potential later use

def restore_original_repr():
	"""
	Restore the original NumPy array __repr__ method.
	"""
	if hasattr(np.ndarray, '_original_repr'):
		np.ndarray.__repr__ = np.ndarray._original_repr
		print("Original NumPy array display restored")
	else:
		print("No original __repr__ method saved to restore")

class ArrayPrintContext:
	"""
	Context manager for temporarily changing numpy print options and __repr__.

	Example:
		with ArrayPrintContext(precision=2):
			print(my_array)  # Will print with 2 decimal places
	"""
	def __init__(self, precision=3, suppress=True, threshold=50, linewidth=100):
		self.precision = precision
		self.suppress = suppress
		self.threshold = threshold
		self.linewidth = linewidth
		self.old_repr = None
		self.old_options = None

	def __enter__(self):
		# Save current settings
		self.old_options = np.get_printoptions()
		self.old_repr = np.ndarray.__repr__

		# Apply temporary settings
		np.set_printoptions(precision=self.precision, suppress=self.suppress,
						   threshold=self.threshold, linewidth=self.linewidth)

		# Apply temporary __repr__
		np.ndarray.__repr__ = lambda self: np.array2string(
			self, precision=self.precision, suppress_small=self.suppress,
			threshold=self.threshold, max_line_width=self.linewidth
		)

		return self

	def __exit__(self, exc_type, exc_val, exc_tb):
		# Restore original settings
		np.set_printoptions(**self.old_options)
		np.ndarray.__repr__ = self.old_repr

# Test the module if run directly
if __name__ == "__main__":
	# Create some test arrays
	arr1d = np.array([1.23456789, 2.3456789, 3.456789])
	arr2d = np.random.random((3, 4))
	arr3d = np.random.random((2, 3, 4))

	# Test default formatting
	print("Default numpy display:")
	print(arr1d)
	print(arr2d)

	# Test custom formatting function
	print("\nCustom formatting function:")
	print(format_array(arr1d))
	print(format_array(arr2d, precision=1))

	# Test setting custom __repr__
	print("\nSetting custom __repr__:")
	set_custom_repr(precision=2)
	print(arr1d)
	print(arr2d)

	# Test context manager
	print("\nUsing context manager:")
	restore_original_repr()  # First restore original

	with ArrayPrintContext(precision=1):
		print("Inside context with precision=1:")
		print(arr1d)
		print(arr2d)

	print("Outside context (original format):")
	print(arr1d)
	print(arr2d)

