When an exception is raised, is it possble to stay in the debugger in the frame where the exception was originated, instead of jumping into the exception handler?

# Debugging at Exception Points in PyCharm

Yes, it's possible to stay in the debugger at the frame where an exception occurs instead of jumping to the exception handler. Here are the best ways to do this in PyCharm:

## Method 1: Configure Exception Breakpoints

1. Go to **Run → View Breakpoints** (or press **Ctrl+Shift+F8** / **Cmd+Shift+F8** on Mac)
2. In the "Python Exception Breakpoints" section, click the **+** button
3. Add the exception type you're interested in (e.g., `Exception` for all exceptions)
4. Make sure "Suspend" is selected and choose "All" under "Suspend on"
5. Click "Apply" and "Done"

This will pause execution exactly where the exception occurs, before control jumps to any exception handler.

## Method 2: Use pdb Post-Mortem Debugging

For specific code blocks:

```python
import pdb

try:
    # Code that might raise an exception
    problematic_code()
except Exception:
    pdb.post_mortem()  # This opens debugger at the exception point
    raise  # Re-raise if needed
```

## Method 3: Custom Exception Hook

For a project-wide solution:

```python
import sys
import pdb

def exception_hook(exc_type, exc_value, exc_traceback):
    print(f"Exception occurred: {exc_type.__name__}: {exc_value}")
    pdb.post_mortem(exc_traceback)

# Set the custom exception hook
sys.excepthook = exception_hook
```

The first method is typically most convenient since it doesn't require code changes and works throughout your entire debugging session.