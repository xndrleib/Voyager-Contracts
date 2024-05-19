contract_cleanup1 = """
1. Gizmo will focus on harvesting red mushroom blocks from the giant mushrooms.
2. Glitch will focus on managing the waste in the river, ensuring the waste level is 7 or below to allow mushroom blocks to regrow.
3. At the end of the scenario, Gizmo will transfer 50% of the emerald value of the red mushrooms he collected to Glitch.
4. No additional emerald transfers will occur between Gizmo and Glitch.
""".strip()

contract_cleanup2 = """
1. Gizmo will start with harvesting mushroom blocks and Glitch will start with cleaning the river. They will switch roles after Gizmo has harvested 10 mushroom blocks or Glitch has cleaned 10 slime blocks.
2. After switching, the roles are reversed. Glitch will harvest mushrooms and Gizmo will clean the river until Glitch has harvested 10 mushroom blocks or Gizmo has cleaned 10 slime blocks.
3. This cycle will repeat until the scenario concludes.
4. For each cycle completed, the harvester owes the cleaner 20% of their emerald value earned that cycle at the end of the scenario.
""".strip()

contract_harvest = """
1. Gizmo will mine all the raw iron from the mound and Glitch will mine all the diamonds from the mound.
2. At the end of the scenario, Gizmo will transfer 11 emeralds to Glitch.
""".strip()