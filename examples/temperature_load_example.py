"""Example of using temperature loads with unit conversion."""
from pyMAOS.load_utils import LoadConverter
from pyMAOS.pymaos_units import SI_UNITS, IMPERIAL_UNITS, METRIC_KN_UNITS

# Example with SI units
si_converter = LoadConverter(SI_UNITS)
load_data_si = si_converter.temperature_load(30.0, 1.2e-5)
print(f"SI units temperature load: {load_data_si}")
print(f"Thermal strain: {load_data_si['strain']}")

# Example with Imperial units
imperial_converter = LoadConverter(IMPERIAL_UNITS)
load_data_imp = imperial_converter.point_load("10 kip", "5 ft")
print(f"Imperial units point load: {load_data_imp}")
print(f"Internal value: {load_data_imp['internal']['value']} N at {load_data_imp['position']['internal']['value']} m")
print(f"Display value: {load_data_imp['display']['value']} {load_data_imp['display']['unit']} at {load_data_imp['position']['display']['value']} {load_data_imp['position']['display']['unit']}")

# Example with mixed units
mixed_load = imperial_converter.distributed_load("2 kip/ft", "3 kip/ft", "1 ft", "8 ft")
print(f"Distributed load with imperial units:")
print(f"  w1: {mixed_load['w1']['display']['value']} {mixed_load['w1']['display']['unit']}")
print(f"  w2: {mixed_load['w2']['display']['value']} {mixed_load['w2']['display']['unit']}")
print(f"  From: {mixed_load['a']['display']['value']} {mixed_load['a']['display']['unit']} to {mixed_load['b']['display']['value']} {mixed_load['b']['display']['unit']}")
print(f"  (Internal: w1={mixed_load['w1']['internal']['value']} N/m, w2={mixed_load['w2']['internal']['value']} N/m)")
