import unittest
, SI_UNITS, IMPERIAL_UNITS, METRIC_KN_UNITS

class TestUnitConversions(unittest.TestCase):
    def test_force_conversions(self):
        # Test converting between force units
        self.assertAlmostEqual(unit_manager.convert_value(1000, 'N', 'kN'), 1.0)
        self.assertAlmostEqual(unit_manager.convert_value(1, 'kN', 'N'), 1000.0)
        self.assertAlmostEqual(unit_manager.convert_value(4.4482, 'N', 'lbf'), 1.0, places=4)
        
    def test_length_conversions(self):
        # Test converting between length units
        self.assertAlmostEqual(unit_manager.convert_value(1, 'm', 'ft'), 3.28084, places=5)
        self.assertAlmostEqual(unit_manager.convert_value(12, 'in', 'ft'), 1.0, places=5)
        
    def test_unit_system_changes(self):
        # Test that changing unit systems updates global variables
        
        # Save original system
        original_force = unit_manager.get_current_units().get("force", "N")
        
        # Change to imperial
        unit_manager.set_display_unit_system(IMPERIAL_UNITS, "imperial")
        self.assertEqual(unit_manager.get_current_units().get("force", ""), "lbf")
        
        # Change to SI
        unit_manager.set_display_unit_system(SI_UNITS, "SI")
        self.assertEqual(unit_manager.get_current_units().get("force", ""), "N")
        
        # Change to metric kN
        unit_manager.set_display_unit_system(METRIC_KN_UNITS, "metric_kN")
        self.assertEqual(unit_manager.get_current_units().get("force", ""), "kN")
        
        # Restore original
        from pyMAOS.pymaos_units import FORCE_UNIT
        self.assertEqual(FORCE_UNIT, "kN")

if __name__ == "__main__":
    unittest.main()
