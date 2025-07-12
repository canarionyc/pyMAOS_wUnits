# 1. Define individual load cases
from pyMAOS.loadcombos import LoadCombo


loadcases = ["D", "L", "W", "S", "E"]  # Dead, Live, Wind, Snow, Earthquake

# 2. Create different load combinations
# Format: LoadCombo(name, factor_dictionary, cases_list, is_ultimate, limit_state_type)

# Serviceability combinations
service_D = LoadCombo("SER_D", {"D": 1.0}, ["D"], False, "SLS")
service_DL = LoadCombo("SER_DL", {"D": 1.0, "L": 1.0}, ["D", "L"], False, "SLS")
service_DLW = LoadCombo("SER_DLW", {"D": 1.0, "L": 1.0, "W": 0.6}, ["D", "L", "W"], False, "SLS")

# Ultimate strength combinations
ultimate_D = LoadCombo("ULT_D", {"D": 1.2}, ["D"], True, "ULS")
ultimate_DL = LoadCombo("ULT_DL", {"D": 1.2, "L": 1.6}, ["D", "L"], True, "ULS") 
ultimate_DLW = LoadCombo("ULT_DLW", {"D": 1.2, "L": 1.0, "W": 1.6}, ["D", "L", "W"], True, "ULS")
ultimate_DLE = LoadCombo("ULT_DLE", {"D": 1.2, "L": 0.5, "E": 1.0}, ["D", "L", "E"], True, "ULS")

# 3. Store combinations in a list for processing
load_combos = [service_D, service_DL, service_DLW, ultimate_D, ultimate_DL, ultimate_DLW, ultimate_DLE]

# 4. Analyze structure for each load combination
results = {}
for combo in load_combos:
    print(f"\nAnalyzing for load combination: {combo.name}")
    results[combo.name] = Structure.solve_linear_static(combo, **vars(args))
    
    # Print results for this combination
    print(f"\nResults for {combo.name}:")
    print("Displacements:")
    for node in node_list:
        tx = node.displacements[combo.name]
        print(f"N{node.uid} -- Ux: {tx[0]:.4E} -- Uy:{tx[1]:.4E} -- Rz:{tx[2]:.4E}")
    
    # Print more results as needed
    # ...
