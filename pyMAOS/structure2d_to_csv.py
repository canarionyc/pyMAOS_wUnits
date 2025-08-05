def export_results_to_csv(self, csv_dir, loadcombos=None):
            """
            Export analysis results to multiple CSV files in the specified directory.

            Parameters
            ----------
            csv_dir : str
                Directory path where CSV files will be stored
            loadcombos : list, optional
                List of load combinations to export results for. If None, uses all available.
            """
            import os
            import csv
            import re

            # Create directory if it doesn't exist
            if not os.path.exists(csv_dir):
                os.makedirs(csv_dir)

            print(f"Exporting results to CSV files in: {csv_dir}")

            # Use all load combos if none specified
            if loadcombos is None:
                # Determine available load combos from node displacements
                if hasattr(self.nodes[0], 'displacements') and self.nodes[0].displacements:
                    available_combos = list(self.nodes[0].displacements.keys())
                    loadcombos = available_combos
                else:
                    print("Warning: No load combinations found in results.")
                    return

            # Helper function to sanitize combo name for filename
            def sanitize_filename(combo):
                # Get a clean identifier from the combo object
                if hasattr(combo, 'name'):
                    # Use the name attribute if available
                    combo_id = str(combo.name)
                else:
                    # Otherwise use string representation but clean it up
                    combo_id = str(combo)

                # Remove invalid filename characters
                # Replace common invalid chars with underscores
                combo_id = re.sub(r'[\\/*?:"<>|\n\r\t]', '_', combo_id)
                # Remove any remaining problematic characters
                combo_id = ''.join(c for c in combo_id if c.isprintable() and c not in '\\/*?:"<>|')
                # Trim to reasonable length and ensure it's not empty
                combo_id = combo_id[:50] or 'unnamed_combo'

                print(f"Sanitized combo name: '{combo}' -> '{combo_id}'")
                return combo_id

            # Export structure summary
            with open(os.path.join(csv_dir, "structure_summary.csv"), 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Property", "Value"])
                writer.writerow(["Total Nodes", self.NJ])
                writer.writerow(["Total Members", self.NM])
                writer.writerow(["Degrees of Freedom", self.NDOF])
                writer.writerow(["Dimensions", self.DIM])
                writer.writerow(["Number of Restraints", self.NR])
                writer.writerow(["Analysis Status", "Unstable" if self._unstable else "Stable"])
                print(f"Exported structure summary to: structure_summary.csv")

            # Export node data
            with open(os.path.join(csv_dir, "nodes.csv"), 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Node ID", "X (m)", "Y (m)", "Restraint X", "Restraint Y", "Restraint Z"])
                for node in self.nodes:
                    rx, ry, rz = node.restraints if hasattr(node, 'restraints') else [0, 0, 0]
                    writer.writerow([node.uid, node.x, node.y, rx, ry, rz])
                print(f"Exported {len(self.nodes)} nodes to: nodes.csv")

            # Export member data
            with open(os.path.join(csv_dir, "members.csv"), 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Member ID", "Node i", "Node j", "Length (m)", "Material", "Section"])
                for member in self.members:
                    writer.writerow([
                        member.uid,
                        member.inode.uid,
                        member.jnode.uid,
                        member.length,
                        member.material.uid if hasattr(member, 'material') else "N/A",
                        member.section.uid if hasattr(member, 'section') else "N/A"
                    ])
                print(f"Exported {len(self.members)} members to: members.csv")

            # Export displacements for each load combo
            for combo in loadcombos:
                combo_id = sanitize_filename(combo)
                filename = f"displacements_{combo_id}.csv"
                with open(os.path.join(csv_dir, filename), 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(["Node ID", "Ux (m)", "Uy (m)", "Rz (rad)"])
                    for node in self.nodes:
                        if hasattr(node, 'displacements') and combo in node.displacements:
                            d = node.displacements[combo]
                            writer.writerow([node.uid, d[0], d[1], d[2]])
                        else:
                            writer.writerow([node.uid, "N/A", "N/A", "N/A"])
                print(f"Exported node displacements for load combo {combo_id} to: {filename}")

            # Export member forces for each load combo
            for combo in loadcombos:
                combo_id = sanitize_filename(combo)
                filename = f"member_forces_{combo_id}.csv"
                with open(os.path.join(csv_dir, filename), 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(["Member ID", "Axial Force i (N)", "Shear i (N)", "Moment i (N·m)",
                                    "Axial Force j (N)", "Shear j (N)", "Moment j (N·m)"])
                    for member in self.members:
                        if hasattr(member, 'member_forces') and combo in member.member_forces:
                            forces = member.member_forces[combo]
                            writer.writerow([
                                member.uid,
                                forces[0], forces[1], forces[2],
                                forces[3], forces[4], forces[5]
                            ])
                        else:
                            writer.writerow([member.uid, "N/A", "N/A", "N/A", "N/A", "N/A", "N/A"])
                print(f"Exported member forces for load combo {combo_id} to: {filename}")

            # Export reactions for each load combo
            for combo in loadcombos:
                combo_id = sanitize_filename(combo)
                filename = f"reactions_{combo_id}.csv"
                with open(os.path.join(csv_dir, filename), 'w', newline='') as f:
                    writer = csv.writer(f)
                    writer.writerow(["Node ID", "Rx (N)", "Ry (N)", "Mz (N·m)"])
                    for node in self.nodes:
                        if hasattr(node, 'reactions') and combo in node.reactions:
                            r = node.reactions[combo]
                            writer.writerow([node.uid, r[0], r[1], r[2]])
                        else:
                            writer.writerow([node.uid, "N/A", "N/A", "N/A"])
                print(f"Exported reactions for load combo {combo_id} to: {filename}")

            print(f"Results exported successfully to {len(loadcombos) * 3 + 3} CSV files")
            return True