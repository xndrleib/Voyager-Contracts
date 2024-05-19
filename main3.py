import argparse
from voyager import MultiAgentVoyager
import time
from api_keys import openai_api_key
from datetime import datetime
import voyager.utils as U

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

multi_options = {
    'scenario_file': "cleanup.json",
    'continuous': True,
    'episode_timeout': 120,
    'num_episodes': 1,
    'negotiator_model_name': model,
    'negotiator_temperature': 0.7,
    'options': options
}

start_time = time.time()

save_dir = f"saves/cleanup/baseline"
U.f_mkdir(save_dir)

contract = "None"

for game in range(5, 20):
    multi_agent = MultiAgentVoyager(
        **multi_options,
        contract_mode="manual",
        contract=contract,
        save_dir=f"{save_dir}/game{game}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    )
    multi_agent.run()
    multi_agent.close()

print(f"Contract {contract} completed. {time.time() - start_time} seconds elapsed.")
