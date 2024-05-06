import argparse
from voyager import MultiAgentVoyager
import time
from api_keys import openai_api_key
from datetime import datetime
import voyager.utils as U
import logging

# Argument parser
parser = argparse.ArgumentParser(description='Running Voyager with different sets of parameters.')
parser.add_argument('--port', type=int, required=True, help='MC port number')
parser.add_argument('--server_port', type=int, default=3000, help='Server port number (default: 3000)')
args = parser.parse_args()

azure_login = None
mc_port = args.port
server_port = args.server_port

model = "gpt-4-turbo"  # "gpt-4-0613" | "gpt-4-turbo" | gpt-3.5-turbo

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

n_games = 1
n_contracts = 10
n_negotiation_tries = 3

start_time = time.time()

for contract_i in range(1, n_contracts+1):
    save_dir = f"saves/cleanup/contract{contract_i}"
    U.f_mkdir(save_dir)

    contract_env = MultiAgentVoyager(**multi_options,
                                     contract_mode="auto",
                                     save_dir=f"{save_dir}/contract_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    contract_env.load_scenario(reset='hard')

    for i in range(n_negotiation_tries):
        try:
            contract_env.negotiate_contract()
            break
        except Exception as e:
            logging.exception(f"Negotiation failed:\n{str(e)}")

    if contract_env.contract is None:
        raise Exception(f'Contract negotiation failed after {n_negotiation_tries} tries')

    contract = contract_env.contract
    contract_env.close()

    for game in range(n_games):
        multi_agent = MultiAgentVoyager(
            **multi_options,
            contract_mode="manual",
            contract=contract,
            save_dir=f"{save_dir}/game{game}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        )
        multi_agent.run()
        multi_agent.close()

    logging.info(f"Contract {contract} completed. {time.time() - start_time} seconds elapsed.")
