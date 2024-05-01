import os.path
import time
import warnings
from typing import Any, Tuple, Dict

import requests
import json

import gymnasium as gym
from gymnasium.core import ObsType

import voyager.utils as U

from .minecraft_launcher import MinecraftInstance
from .process_monitor import SubprocessMonitor


class VoyagerEnv(gym.Env):
    def __init__(
        self,
        mc_port=None,
        username="bot",
        azure_login=None,
        server_host="http://127.0.0.1",
        server_port=3000,
        request_timeout=600,
        log_path="./logs",
    ):
        if not mc_port and not azure_login:
            raise ValueError("Either mc_port or azure_login must be specified")
        if mc_port and azure_login:
            warnings.warn(
                "Both mc_port and mc_login are specified, mc_port will be ignored"
            )
        self.mc_port = mc_port
        self.username = username
        self.azure_login = azure_login
        self.server = f"{server_host}:{server_port}"
        self.server_port = server_port
        self.request_timeout = request_timeout
        self.log_path = log_path
        self.mineflayer = self.get_mineflayer_process(server_port)
        if azure_login:
            self.mc_instance = self.get_mc_instance()
        else:
            self.mc_instance = None
        self.has_reset = False
        self.reset_options = None
        self.connected = False
        self.server_paused = False

    def get_mineflayer_process(self, server_port):
        U.f_mkdir(self.log_path, "mineflayer")
        file_path = os.path.abspath(os.path.dirname(__file__))
        return SubprocessMonitor(
            commands=[
                "node",
                U.f_join(file_path, "mineflayer/index.js"),
                str(server_port),
            ],
            name="mineflayer",
            ready_match=r"Server started on port (\d+)",
            log_path=U.f_join(self.log_path, "mineflayer"),
        )

    def get_mc_instance(self):
        print("Creating Minecraft server")
        U.f_mkdir(self.log_path, "minecraft")
        return MinecraftInstance(
            **self.azure_login,
            mineflayer=self.mineflayer,
            log_path=U.f_join(self.log_path, "minecraft"),
        )

    def send_request(self, url, data, timeout=None):
        try:
            response = requests.post(url, json=data, timeout=timeout or self.request_timeout)
            response.raise_for_status()  # Raises an HTTPError for bad responses
            return response.json()
        except requests.exceptions.HTTPError as errh:
            print("HTTP Error:", errh)
        except requests.exceptions.ConnectionError as errc:
            print("Error Connecting:", errc)
        except requests.exceptions.Timeout as errt:
            print("Timeout Error:", errt)
        except requests.exceptions.RequestException as err:
            print("Oops: Something Else", err)
        return None

    def start_mc_instance(self):
        print("Starting Minecraft server")
        self.mc_instance.run()
        self.mc_port = self.mc_instance.port
        self.reset_options["port"] = self.mc_instance.port
        print(f"Server started on port {self.reset_options['port']}")

    def restart_mineflayer_with_backoff(self):
        retry, max_retries, backoff_factor = 0, 3, 2
        while retry <= max_retries:
            print("Mineflayer process has exited, restarting")
            self.mineflayer.run()
            time.sleep(backoff_factor ** retry)
            if self.mineflayer.is_running:
                return
            retry += 1
        raise RuntimeError("Failed to restart Mineflayer after several attempts")

    def try_server_start_endpoint(self):
        for retry in range(3):
            result = self.send_request(f"{self.server}/start", self.reset_options, timeout=10)
            if result is not None:
                print("Server started successfully")
                return result
            print(f"Attempt {retry + 1}: bot start failed, retrying...")
        raise RuntimeError("Failed to start server via /start endpoint")

    def check_process(self):
        if self.mc_instance and not self.mc_instance.is_running:
            self.start_mc_instance()

        if not self.mineflayer.is_running:
            self.restart_mineflayer_with_backoff()

        return self.try_server_start_endpoint()

    def step(self, code: str, programs: str = ""):
        if not self.has_reset:
            raise RuntimeError("Environment has not been reset yet")
        self.check_process()
        data = {"code": code, "programs": programs}
        result = self.send_request(f"{self.server}/step", data)
        if result is None:
            raise RuntimeError("Failed to get a valid response from the Minecraft server")
        return json.loads(result)

    def render(self):
        raise NotImplementedError("render is not implemented")

    def reset(self, *, seed=None, options=None) -> Tuple[ObsType, Dict[str, Any]]:
        if options is None:
            options = {}

        # Validate the options dictionary to ensure correct settings are applied
        if options.get("inventory", {}) and options.get("mode", "hard") != "hard":
            raise RuntimeError("Inventory can only be set when the reset mode is 'hard'.")

        # Setting up reset options based on the provided arguments and defaults
        self.reset_options = {
            "port": self.mc_port,
            "username": self.username,
            "reset": options.get("mode", "hard"),
            "inventory": options.get("inventory", {}),
            "equipment": options.get("equipment", []),
            "spread": options.get("spread", False),
            "waitTicks": options.get("wait_ticks", 5),
            "position": options.get("position", None),
        }

        # Ensure mineflayer is properly shutdown before attempting a reset
        self.mineflayer.stop()
        time.sleep(1)  # Ensures the process has time to terminate properly

        # Re-initialize or check server/mineflayer process status
        returned_data = self.check_process()
        if returned_data is None:
            raise RuntimeError("Failed to start or reset the server process properly.")

        self.has_reset = True
        self.connected = True

        # All future resets after the initial setup will be of type 'soft'
        self.reset_options["reset"] = "soft"

        return json.loads(returned_data)

    def close(self):
        print('close')
        # self.unpause()
        if self.connected:
            res = requests.post(f"{self.server}/stop")
            if res.status_code == 200:
                self.connected = False
        if self.mc_instance:
            self.mc_instance.stop()
        self.mineflayer.stop()
        return not self.connected

    def pause(self):
        if self.mineflayer.is_running and not self.server_paused:
            res = requests.post(f"{self.server}/pause")
            if res.status_code == 200:
                self.server_paused = True
        return self.server_paused

    def unpause(self):
        if self.mineflayer.is_running and self.server_paused:
            res = requests.post(f"{self.server}/pause")
            if res.status_code == 200:
                self.server_paused = False
            else:
                print(res.json())
        return self.server_paused
