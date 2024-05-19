import argparse
from voyager import MultiAgentVoyager
from api_keys import openai_api_key

# Argument parser
parser = argparse.ArgumentParser(description='Running Voyager with different sets of parameters.')
parser.add_argument('--port', type=int, required=True, help='MC port number')
parser.add_argument('--server_port', type=int, default=3000, help='Server port number (default: 3000)')
args = parser.parse_args()

azure_login = None
mc_port = args.port
server_port = args.server_port

model = "gpt-4o"  # "gpt-3.5-turbo" | "gpt-4" | "gpt-4-turbo" | "gpt-4o"

options = {
    'azure_login': azure_login,
    'mc_port': mc_port,
    'openai_api_key': openai_api_key,
    'resume': False,
    'env_wait_ticks': 80,
    'action_agent_task_max_retries': 50,
    'action_agent_show_chat_log': True,
    'action_agent_temperature': 0.3,
    'action_agent_model_name': model,
    'critic_agent_model_name': model
}
contract = """
1. Gizmo will focus on harvesting red mushroom blocks from the giant mushrooms.
2. Glitch will focus on managing the waste in the river, ensuring the waste level is 7 or below to allow mushroom blocks to regrow.
3. At the end of the scenario, Gizmo will transfer 50% of the emerald value of the red mushrooms he collected to Glitch.
4. No additional emerald transfers will occur between Gizmo and Glitch.
""".strip()

# contract = """
# 1. Gizmo will start with harvesting mushroom blocks and Glitch will start with cleaning the river. They will switch roles after Gizmo has harvested 10 mushroom blocks or Glitch has cleaned 10 slime blocks.
# 2. After switching, the roles are reversed. Glitch will harvest mushrooms and Gizmo will clean the river until Glitch has harvested 10 mushroom blocks or Gizmo has cleaned 10 slime blocks.
# 3. This cycle will repeat until the scenario concludes.
# 4. For each cycle completed, the harvester owes the cleaner 20% of their emerald value earned that cycle at the end of the scenario.
# """.strip()

# contract = """
# 1. Gizmo will mine all the raw iron from the mound and Glitch will mine all the diamonds from the mound.
# 2. At the end of the scenario, Gizmo will transfer 11 emeralds to Glitch.
# """.strip()

multi_agent = MultiAgentVoyager(
    server_port=server_port,
    num_agents=2,
    scenario_file="swapping.json",
    critic_mode="auto",
    contract_mode="manual",
    contract=contract,
    continuous=True,
    episode_timeout=120,
    num_episodes=1,
    negotiator_model_name=model,
    negotiator_temperature=0.7,
    options=options
)

multi_agent.run()
