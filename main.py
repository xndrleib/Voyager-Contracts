import argparse
from voyager import MultiAgentVoyager
from api_keys import openai_api_key
import contract_examples

# Argument parser
parser = argparse.ArgumentParser(description='Running Voyager with different sets of parameters.')
parser.add_argument('--port', type=int, required=True, help='MC port number')
parser.add_argument('--server_port', type=int, default=3000, help='Server port number (default: 3000)')
args = parser.parse_args()

azure_login = None
mc_port = args.port
server_port = args.server_port

model = "gpt-4o"  # "gpt-3.5-turbo" | "gpt-4" | "gpt-4-turbo" | "gpt-4o"

scenario_file = 'cleanup.json'  # 'cleanup.json' | 'swapping.json'
contract = contract_examples.contract_cleanup1
num_episodes = 2

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


multi_agent = MultiAgentVoyager(
    server_port=server_port,
    num_agents=2,
    scenario_file=scenario_file,
    critic_mode="auto",
    contract_mode="manual",
    contract=contract,
    continuous=True,
    episode_timeout=90,
    num_episodes=num_episodes,
    negotiator_model_name=model,
    negotiator_temperature=0.7,
    options=options
)

multi_agent.run()
