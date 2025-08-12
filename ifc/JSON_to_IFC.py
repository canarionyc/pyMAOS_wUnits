import json
import uuid
import sys
import os
import datetime
import ifcopenshell
import ifcopenshell.geom
import ifcopenshell.api
import scipy as sp
import math
import traceback
from ifcopenshell.util import representation

def debug_print(*args, **kwargs):
    """Helper function for debug printing"""
    print(*args, **kwargs)
    sys.stdout.flush()

def create_guid():
    """Create a GUID for IFC entities"""
    return ifcopenshell.guid.compress(uuid.uuid4().hex)

def convert_json_to_ifc(json_file):
    """Convert JSON structure to IFC format"""
    debug_print(f"Converting {json_file} to IFC")

    # Read the JSON file
    with open(json_file, 'r') as f:
        data = json.load(f)

    # Extract model name
    model_name = data.get("design_parameters", {}).get("name", "Structural Model")
    author = data.get("design_parameters", {}).get("author", "Converter")
    description = data.get("design_parameters", {}).get("description", "")

    # Create a new IFC file
    timestamp = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
    filename = os.path.splitext(os.path.basename(json_file))[0]
    ifc_file = ifcopenshell.file()

    # Set up the IFC file header
    debug_print("Setting up IFC header")
    ifc_file.wrapped_data.header.file_name.name = f"{filename}.ifc"
    ifc_file.wrapped_data.header.file_description.description = (description,)
    ifc_file.wrapped_data.header.file_name.time_stamp = timestamp
    ifc_file.wrapped_data.header.file_name.author = (author,)
    ifc_file.wrapped_data.header.file_name.organization = ("",)

    # Create IFC project structure
    debug_print("Creating project structure")
    project = ifcopenshell.api.run("root.create_entity", ifc_file, ifc_class="IfcProject", name=model_name)

    # Define units
    length_unit = data.get("units", {}).get("length", "ft")
    force_unit = data.get("units", {}).get("force", "kip")
    pressure_unit = data.get("units", {}).get("pressure", "ksi")

    # Create units
    debug_print("Setting up units")
    unit_assignments = ifcopenshell.api.run("unit.assign_unit", ifc_file)

    # Length unit - handle imperial units correctly
    if length_unit == "ft":
        # Create a conversion-based unit for feet (1 foot = 0.3048 meters)
        length_si_unit = ifcopenshell.api.run("unit.add_si_unit", ifc_file,
                                            unit_type="LENGTHUNIT")
        ifcopenshell.api.run("unit.add_conversion_based_unit", ifc_file,
                           name="foot", unit_type="LENGTHUNIT",
                           conversion_factor=0.3048, converted_unit=length_si_unit)
    elif length_unit == "in":
        # Create a conversion-based unit for inches (1 inch = 0.0254 meters)
        length_si_unit = ifcopenshell.api.run("unit.add_si_unit", ifc_file,
                                            unit_type="LENGTHUNIT")
        ifcopenshell.api.run("unit.add_conversion_based_unit", ifc_file,
                           name="inch", unit_type="LENGTHUNIT",
                           conversion_factor=0.0254, converted_unit=length_si_unit)
    else:
        # Default to meters
        ifcopenshell.api.run("unit.add_si_unit", ifc_file, unit_type="LENGTHUNIT")

    # Force unit - create appropriate conversion
    if force_unit == "kip":
        # 1 kip = 4448.2216 newtons
        force_si_unit = ifcopenshell.api.run("unit.add_si_unit", ifc_file,
                                           unit_type="FORCEUNIT")
        ifcopenshell.api.run("unit.add_conversion_based_unit", ifc_file,
                           name="kip", unit_type="FORCEUNIT",
                           conversion_factor=4448.2216, converted_unit=force_si_unit)
    else:
        # Default to newtons
        ifcopenshell.api.run("unit.add_si_unit", ifc_file, unit_type="FORCEUNIT")

    # Pressure unit - create appropriate conversion
    if pressure_unit == "ksi":
        # 1 ksi = 6894757.2932 pascals
        pressure_si_unit = ifcopenshell.api.run("unit.add_si_unit", ifc_file,
                                              unit_type="PRESSUREUNIT")
        ifcopenshell.api.run("unit.add_conversion_based_unit", ifc_file,
                           name="ksi", unit_type="PRESSUREUNIT",
                           conversion_factor=6894757.2932, converted_unit=pressure_si_unit)
    else:
        # Default to pascals
        ifcopenshell.api.run("unit.add_si_unit", ifc_file, unit_type="PRESSUREUNIT")

    # Create a site, building, and building story
    site = ifcopenshell.api.run("root.create_entity", ifc_file,
                              ifc_class="IfcSite", name="Site")
    ifcopenshell.api.run("aggregate.assign_object", ifc_file,
                       products=[site], relating_object=project)

    building = ifcopenshell.api.run("root.create_entity", ifc_file,
                                  ifc_class="IfcBuilding", name="Building")
    ifcopenshell.api.run("aggregate.assign_object", ifc_file,
                       products=[building], relating_object=site)

    story = ifcopenshell.api.run("root.create_entity", ifc_file,
                               ifc_class="IfcBuildingStorey", name="Story")
    ifcopenshell.api.run("aggregate.assign_object", ifc_file,
                       products=[story], relating_object=building)

    # Create context for geometry representation
    debug_print("Creating geometry context")
    context = ifcopenshell.api.run("context.add_context", ifc_file,
                                 context_type="Model")
    body_context = ifcopenshell.api.run("context.add_context", ifc_file,
                                      context_type="Model",
                                      context_identifier="Body",
                                      target_view="MODEL_VIEW",
                                      parent=context)

    # Create materials dictionary
    debug_print("Creating materials")
    materials_dict = {}
    for material_data in data.get("materials", []):
        material_id = material_data["id"]
        material_name = f"Material_{material_id}"

        material = ifcopenshell.api.run("material.add_material", ifc_file,
                                      name=material_name)

        # Set material properties if available
        if "type" in material_data:
            if material_data["type"] == "isotropic":
                try:
                    # Create property set manually
                    pset_name = "Pset_MaterialMechanical"
                    young_modulus = float(material_data.get("E", 0))
                    shear_modulus = float(material_data.get("G", 0))
                    poisson_ratio = float(material_data.get("nu", 0))

                    props = {
                        "YoungModulus": young_modulus,
                        "ShearModulus": shear_modulus,
                        "PoissonRatio": poisson_ratio
                    }

                    if "rho" in material_data:
                        props["MassDensity"] = float(material_data["rho"])

                    # Create property set manually
                    pset = ifc_file.create_entity(
                        "IfcPropertySet",
                        GlobalId=create_guid(),
                        Name=pset_name,
                        Description=f"Material properties for {material_name}",
                        HasProperties=[]
                    )

                    # Create and add properties to the property set
                    for prop_name, prop_value in props.items():
                        # Ensure value is a proper float
                        if prop_value is not None:
                            single_prop = ifc_file.create_entity(
                                "IfcPropertySingleValue",
                                Name=prop_name,
                                NominalValue=ifc_file.create_entity("IfcReal", prop_value)
                            )
                            pset.HasProperties = list(pset.HasProperties) + [single_prop]

                    # Create material property relationship
                    rel = ifc_file.create_entity(
                        "IfcRelDefinesByProperties",
                        GlobalId=create_guid(),
                        RelatedObjects=[material],
                        RelatingPropertyDefinition=pset
                    )

                    debug_print(f"Created material properties for {material_name}")
                except Exception as e:
                    debug_print(f"Warning: Could not set material properties: {e}")
                    pass

        materials_dict[material_id] = material

    # Create section profiles
    debug_print("Creating section profiles")
    section_dict = {}
    for section_data in data.get("sections", []):
        section_id = section_data["id"]
        area = float(section_data.get("area", 0))
        radius = float(section_data.get("r", 0))

        # For simplicity, we'll create circular profiles for all sections
        # In a full implementation, you would determine the actual profile shape
        profile = ifc_file.create_entity(
            "IfcCircleProfileDef",
            ProfileType="AREA",
            ProfileName=f"Section_{section_id}",
            Radius=radius
        )

        section_dict[section_id] = profile

    # Create nodes
    debug_print("Creating nodes")
    node_dict = {}
    for node_data in data.get("nodes", []):
        node_id = node_data["id"]
        x = float(node_data["x"])
        y = float(node_data["y"])
        z = float(node_data["z"])

        # Create a cartesian point for each node
        point = ifc_file.create_entity(
            "IfcCartesianPoint",
            Coordinates=(x, y, z)
        )

        node_dict[node_id] = point

    # Create members
    debug_print("Creating structural members")
    member_dict = {}
    for member_data in data.get("members", []):
        member_id = member_data["id"]
        i_node_id = member_data["i_node"]
        j_node_id = member_data["j_node"]
        material_id = member_data.get("material", 1)
        section_id = member_data.get("section", 1)

        # Get the start and end points
        start_point = node_dict[i_node_id]
        end_point = node_dict[j_node_id]

        # Extract raw coordinates
        start_coords = start_point.Coordinates
        end_coords = end_point.Coordinates

        # Create a properly formatted coordinate list - must be a list of lists of floats
        coord_list = [[float(start_coords[0]), float(start_coords[1]), float(start_coords[2])],
                      [float(end_coords[0]), float(end_coords[1]), float(end_coords[2])]]

        # Create point list properly
        point_list = ifc_file.create_entity(
            "IfcCartesianPointList3D",
            CoordList=coord_list
        )

        # Create a polyline from the points
        polyline = ifc_file.create_entity(
            "IfcPolyline",
            Points=[start_point, end_point]
        )

        # Create extrusion profile
        profile = section_dict.get(section_id)
        if not profile:
            profile = section_dict.get(1, section_dict[list(section_dict.keys())[0]])

        # Calculate member length for extrusion
        p1 = start_point.Coordinates
        p2 = end_point.Coordinates
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        dz = p2[2] - p1[2]
        length = math.sqrt(dx*dx + dy*dy + dz*dz)

        # Create direction for extrusion
        if length > 0:
            direction = ifc_file.create_entity(
                "IfcDirection",
                DirectionRatios=(dx/length, dy/length, dz/length)
            )
        else:
            direction = ifc_file.create_entity(
                "IfcDirection",
                DirectionRatios=(0, 0, 1)
            )

        # Create axis placement
        axis_placement = ifc_file.create_entity(
            "IfcAxis2Placement3D",
            Location=start_point,
            Axis=direction,
            RefDirection=None
        )

        # Create swept solid representation
        swept_area = ifc_file.create_entity(
            "IfcExtrudedAreaSolid",
            SweptArea=profile,
            Position=axis_placement,
            ExtrudedDirection=direction,
            Depth=length
        )

        # Create the shape representation
        shape_representation = ifc_file.create_entity(
            "IfcShapeRepresentation",
            ContextOfItems=body_context,
            RepresentationIdentifier="Body",
            RepresentationType="SweptSolid",
            Items=[swept_area]
        )

        product_shape = ifc_file.create_entity(
            "IfcProductDefinitionShape",
            Representations=[shape_representation]
        )

        # Create the actual beam
        beam = ifc_file.create_entity(
            "IfcBeamStandardCase",
            GlobalId=create_guid(),
            Name=f"Member_{member_id}",
            Description=f"Structural member connecting nodes {i_node_id} and {j_node_id}",
            ObjectPlacement=None,  # Will be set below
            Representation=product_shape
        )

        # Assign material
        material = materials_dict.get(material_id)
        if material:
            material_association = ifc_file.create_entity(
                "IfcMaterialProfileSetUsage",
                ForProfileSet=ifc_file.create_entity(
                    "IfcMaterialProfileSet",
                    MaterialProfiles=[
                        ifc_file.create_entity(
                            "IfcMaterialProfile",
                            Name=f"MP_{member_id}",
                            Material=material,
                            Profile=profile
                        )
                    ]
                )
            )
            ifcopenshell.api.run("material.assign_material", ifc_file,
                               products=[beam],
                               material=material_association)

        # Add beam to building story - using direct entity creation instead of API
        # This replaces the spatial.assign_object API call which is missing
        if hasattr(story, "ContainsElements") and story.ContainsElements:
            # Get existing relationship if it exists
            rel = story.ContainsElements[0]
            # Add this beam to the related elements
            related_elements = list(rel.RelatedElements)
            if beam not in related_elements:
                related_elements.append(beam)
                rel.RelatedElements = related_elements
        else:
            # Create a new relationship
            rel = ifc_file.create_entity(
                "IfcRelContainedInSpatialStructure",
                GlobalId=create_guid(),
                OwnerHistory=None,
                Name=f"Rel {story.Name} - {beam.Name}",
                Description="Spatial containment relationship",
                RelatedElements=[beam],
                RelatingStructure=story
            )

        member_dict[member_id] = beam

    # Create supports
    debug_print("Creating supports")
    for support_data in data.get("supports", []):
        node_id = support_data["node"]
        ux = support_data.get("ux", 0)
        uy = support_data.get("uy", 0)
        uz = support_data.get("uz", 0)
        rx = support_data.get("rx", 0)
        ry = support_data.get("ry", 0)
        rz = support_data.get("rz", 0)

        # Get the support point
        point = node_dict.get(node_id)
        if not point:
            debug_print(f"Warning: Node {node_id} for support not found")
            continue

        # Create a support representation (simplified as a small cube)
        box = ifc_file.create_entity(
            "IfcBoundingBox",
            Corner=point,
            XDim=0.2,
            YDim=0.2,
            ZDim=0.2
        )

        shape_representation = ifc_file.create_entity(
            "IfcShapeRepresentation",
            ContextOfItems=body_context,
            RepresentationIdentifier="Body",
            RepresentationType="BoundingBox",
            Items=[box]
        )

        product_shape = ifc_file.create_entity(
            "IfcProductDefinitionShape",
            Representations=[shape_representation]
        )

        # Create boundary condition values
        # For IFC, we need to create IfcBoolean entities or use numeric stiffness values
        # Using a very high stiffness value for fixed conditions, None for free
        very_high_stiffness = 1e10  # Very high stiffness to simulate fixed condition

        # Helper function to create stiffness value
        def create_stiffness_value(is_fixed):
            if is_fixed:
                # Return a very high stiffness value for fixed condition
                return ifc_file.create_entity("IfcLinearStiffnessMeasure", very_high_stiffness)
            else:
                # Return None for free condition
                return None

        # Create the support as a structural point connection
        support = ifc_file.create_entity(
            "IfcStructuralPointConnection",
            GlobalId=create_guid(),
            Name=f"Support_{node_id}",
            Description=f"Support at node {node_id}",
            ObjectPlacement=None,
            Representation=product_shape,
            AppliedCondition=ifc_file.create_entity(
                "IfcBoundaryNodeCondition",
                Name=f"Constraints_{node_id}",
                TranslationalStiffnessX=create_stiffness_value(ux),
                TranslationalStiffnessY=create_stiffness_value(uy),
                TranslationalStiffnessZ=create_stiffness_value(uz),
                RotationalStiffnessX=ifc_file.create_entity("IfcRotationalStiffnessMeasure", very_high_stiffness) if rx else None,
                RotationalStiffnessY=ifc_file.create_entity("IfcRotationalStiffnessMeasure", very_high_stiffness) if ry else None,
                RotationalStiffnessZ=ifc_file.create_entity("IfcRotationalStiffnessMeasure", very_high_stiffness) if rz else None
            )
        )

        # Add support to building story - using direct entity creation instead of API
        # This replaces the spatial.assign_object API call which is missing
        if hasattr(story, "ContainsElements") and story.ContainsElements:
            # Get existing relationship if it exists
            rel = story.ContainsElements[0]
            # Add this support to the related elements
            related_elements = list(rel.RelatedElements)
            if support not in related_elements:
                related_elements.append(support)
                rel.RelatedElements = related_elements
        else:
            # Create a new relationship
            rel = ifc_file.create_entity(
                "IfcRelContainedInSpatialStructure",
                GlobalId=create_guid(),
                OwnerHistory=None,
                Name=f"Rel {story.Name} - {support.Name}",
                Description="Spatial containment relationship",
                RelatedElements=[support],
                RelatingStructure=story
            )

    # Write the IFC file
    output_file = os.path.splitext(json_file)[0] + ".ifc"
    ifc_file.write(output_file)
    debug_print(f"IFC file saved as {output_file}")

    return output_file

def main():
    if len(sys.argv) < 2:
        print("Usage: python json_to_ifc.py <json_file>")
        return

    json_file = sys.argv[1]
    if not os.path.exists(json_file):
        print(f"Error: File {json_file} not found")
        return

    try:
        ifc_file = convert_json_to_ifc(json_file)
        print(f"Successfully converted {json_file} to {ifc_file}")
    except Exception as e:
        print(f"Error converting file: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()