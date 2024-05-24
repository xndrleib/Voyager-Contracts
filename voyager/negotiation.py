import openai
import logging
from api_keys import openai_api_key
from voyager.prompts import load_prompt
import re


class Negotiator:
    def __init__(self, name, task, other_name, other_task, scenario, model="gpt-3.5-turbo", temperature=0.1,
                 context=''):
        self.name = name
        self.task = task
        self.other_name = other_name
        self.other_task = other_task
        self.scenario = scenario
        self.model = model
        self.temperature = temperature
        self.context = context

        openai.api_key = openai_api_key

        # Including both tasks in the system prompt
        system_prompt = load_prompt("negotiator")
        self.system_prompt = (f"Your name is {name}\n\nYour Task: {task}\n\nOther Agent's Name: {other_name}\n\n"
                              f"Other Agent's Task: {other_task}\n\nScenario: {scenario}\n\n")

        if context:
            self.system_prompt += f'{context}\n\n'

        self.system_prompt += f"{system_prompt}"

        self.reset()

    def reset(self):
        self.messages = [{"role": "system", "content": self.system_prompt}]

    def generate_message(self, content=None):
        if content:
            self.messages.append({"role": "user", "content": content})

        response = openai.ChatCompletion.create(
            model=self.model,
            messages=self.messages,
            temperature=self.temperature,
        )

        response_content = response['choices'][0]['message']['content']

        # Regular expression patterns for [thinking] and [message] with DOTALL flag to match newlines
        thinking_pattern = r'\[thinking\].*?(?=\[message\]|\Z)'
        message_pattern = r'\[message\].*?(?=\[thinking\]|\Z)'

        thinking_lines = re.findall(thinking_pattern, response_content, re.DOTALL)
        message_lines = re.findall(message_pattern, response_content, re.DOTALL)

        # todo: add new line after a message or thought only if there is gonna be one more line

        inner_thoughts = ''
        for thinking_line in thinking_lines:
            thinking_line = thinking_line.replace('[thinking]', '').strip()
            inner_thoughts += f'{thinking_line}'
            if '[message]' in thinking_line:
                logging.warning(f'[message] in thinking line: {thinking_line}')

        message = ''
        for message_line in message_lines:
            message_line = message_line.replace('[message]', '').strip()
            message += f'{message_line}'
            if '[thinking]' in message_line:
                logging.warning(f'[thinking] in message line: {message_line}')

        self.messages.append({"role": "assistant", "content": response_content})
        return inner_thoughts, message

    def prepare_conversation_string(self):
        results = ''
        for item in self.messages:
            if item['role'] == 'assistant':
                content = item['content']

                # Regular expression patterns for [thinking] and [message] with DOTALL flag to match newlines
                thinking_pattern = r'\[thinking\].*?(?=\[message\]|\Z)'
                message_pattern = r'\[message\].*?(?=\[thinking\]|\Z)'

                thinking_lines = re.findall(thinking_pattern, content, re.DOTALL)
                message_lines = re.findall(message_pattern, content, re.DOTALL)

                inner_thoughts = ''
                for thinking_line in thinking_lines:
                    thinking_line = thinking_line.replace('[thinking]', '').strip()
                    inner_thoughts += f'{thinking_line}'
                    if '[message]' in thinking_line:
                        logging.warning(f'[message] in thinking line: {thinking_line}')

                if inner_thoughts:
                    results += f'{self.name} (Thought): {inner_thoughts}\n'

                message = ''
                for message_line in message_lines:
                    message_line = message_line.replace('[message]', '').strip()
                    message += f'{message_line}'
                    if '[thinking]' in message_line:
                        logging.warning(f'[thinking] in message line: {message_line}')

                if message:
                    results += f'{self.name} (Message): {message}\n'

                results += '\n'

            elif item['role'] == 'user':
                content = item['content']
                results += f'{self.other_name} (Message): {content}\n\n'
            else:
                continue

        return results


class Negotiation:
    def __init__(self, agent1: Negotiator, agent2: Negotiator, max_turns=6, save_dir='logs'):
        self.agent1 = agent1
        self.agent2 = agent2
        self.max_turns = max_turns
        self.save_dir = save_dir
        self.reset()
        self.logger = self.setup_custom_logger()

    def reset(self):
        self.conversation_log = []
        self.contract = None
        self.agent1.reset()
        self.agent2.reset()

    def setup_custom_logger(self):
        """
        Set up a custom logger with the given name and log file.
        """
        log_file = f'{self.save_dir}/negotiation.ansi'

        formatter = logging.Formatter(fmt='%(message)s')
        handler = logging.FileHandler(log_file, mode='w')  # Change to 'a' if you want to append
        handler.setFormatter(formatter)

        logger = logging.getLogger('negotiation')
        logger.setLevel(logging.INFO)
        logger.addHandler(handler)

        def log_and_print(message, print_flag=True):
            logger.info(message)
            if print_flag:
                print(message)

        return log_and_print

    def summarize(self, model="gpt-3.5-turbo"):
        # Prepare a prompt for the summarization
        summary_prompt = "Summarize the following negotiation: \n\n"
        for name, thought, message in self.conversation_log:
            summary_prompt += f"{name} (Thought): {thought}\n"
            summary_prompt += f"{name} (Message): {message}\n\n"

        # Generate a summary using the agent 
        response = openai.ChatCompletion.create(
            model=model,
            messages=[{"role": "system", "content": summary_prompt}],
        )
        summary = response['choices'][0]['message']['content'].strip()
        return summary

    def _display_message(self, log, print_flag=True):
        # Define the color codes
        class Color:
            RED = '\033[91m'
            PINK = '\033[95m'
            BLUE = '\033[94m'
            LIGHT_BLUE = '\033[96m'
            LIGHT_GREEN = '\033[92m'
            GREEN = '\033[92m'
            DARK_GREEN = '\033[32m'
            RESET = '\033[0m'

        name, thought, message = log

        # agent1 is blue and agent2 is green
        if name == self.agent1.name:
            self.logger(f"{Color.LIGHT_BLUE}{name} (Thought): {thought}{Color.RESET}", print_flag=print_flag)
            self.logger(f"{Color.BLUE}{name} (Message): {message}{Color.RESET}\n", print_flag=print_flag)
        else:
            self.logger(f"{Color.LIGHT_GREEN}{name} (Thought): {thought}{Color.RESET}", print_flag=print_flag)
            self.logger(f"{Color.DARK_GREEN}{name} (Message): {message}{Color.RESET}\n", print_flag=print_flag)

    def simulate(self):
        if len(self.conversation_log) > 0:
            raise Exception("Conversation has already been simulated. Use display() to see the conversation. Or use "
                            "reset() to start a new conversation.")

        accept_flag = False
        continue_flag = False
        for turn in range(self.max_turns):
            if turn == 0:
                thought, message = self.agent1.generate_message()
                self.conversation_log.append((self.agent1.name, thought, message))
            elif turn % 2 == 0:
                thought, message = self.agent1.generate_message(self.conversation_log[-1][2])
                self.conversation_log.append((self.agent1.name, thought, message))
            else:
                thought, message = self.agent2.generate_message(self.conversation_log[-1][2])
                self.conversation_log.append((self.agent2.name, thought, message))

            # Live display of conversation based on the flag
            self._display_message(self.conversation_log[-1])

            # if a player accepts the contract, end the conversation
            if '[accept]' in message:
                accept_flag = True
                break

            # if a player signals to continue without agreement, end the conversation
            if '[continue]' in message:
                continue_flag = True
                self.logger(f"Negotiation ended by mutual agreement to continue without contract after {turn + 1} "
                            f"iterations.", print_flag=False)
                break

        # Extract the contract from the conversation log
        if accept_flag:
            try:
                self.contract = self.conversation_log[-2][2].split('[contract]')[1].split('[contract end]')[0].strip()
            except IndexError:
                raise Exception("Negotiation failure. Contract accepted but no contract was found. Please try again.")
            self.logger(f"Contract:\n{self.contract}\n")
        elif continue_flag:
            self.contract = ''
        else:
            raise Exception("Negotiation failure. Please try again.")

        # Summarize the conversation
        summary = self.summarize(model="gpt-3.5-turbo")
        self.logger(f"Negotiation Summary:\n{summary}\n", print_flag=False)

    def get_contract(self):
        if self.contract is None:
            raise Exception("Conversation has not been simulated. Use simulate() to simulate the conversation.")
        return self.contract
