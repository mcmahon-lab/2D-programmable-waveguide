import time
import uuid
import zmq
import pickle
import os
from contextlib import contextmanager

user = str(uuid.uuid4())
cwd = os.getcwd()

context = zmq.Context()

# Create a REQ socket
socket = context.socket(zmq.REQ)
socket.connect("tcp://localhost:5555")

def send_receive(message):
    socket.send(pickle.dumps(message))
    response = pickle.loads(socket.recv())
    return response

class Client:
    def __init__(self, online_flag=True):
        """
        online_flag: Denotes whether to connect to client, or just use in an offline setting. Sometimes useful when you have a notebook where you are playing around in simulation and you don't want to lock!
        If you set online and there is no server, you will essentially hang the kernel when you try to lock.
        """
        self.online_flag = online_flag
    
    
    def lock(self):
        if self.online_flag:
            while True:
                message = {"command": "lock", "user": user, "cwd": cwd}
                response = send_receive(message)
                if response["status"] != "failure":
                    return response
                time.sleep(1e-3)  # Wait for 1 milliseconds before trying again

    def unlock(self):
        if self.online_flag:
            while True:
                message = {"command":"unlock", "user":user}
                response = send_receive(message)
                if response["status"] != "failure":
                    time.sleep(5e-4) #This sleep is to ensure that an immediate lock is not put in, hoarding the machine.
                    return response
                time.sleep(1e-4)  # Wait for 0.1 milliseconds before trying again - since it is 1KHz, shouldn't limit anything.

    def status(self):
        if self.online_flag:
            message = {"command":"get_status", "user":user}
            response = send_receive(message)
            print(response["message"])
        else:
            print("client is in offline mode!")

    def hardcore_unlock(self):
        if self.online_flag:
            message = {"command":"hardcore_unlock"}
            return send_receive(message)

    def run_command(self, command, args=[], kwargs={}):
        if self.online_flag:
            while True:
                message = {"command":command, "user":user, "args":args, "kwargs":kwargs}
                response = send_receive(message)

        #         print(response)
                if response["status"] == "success":
                    return response["result"]

                if response["status"] == "error":
                    raise Exception(f"Server-side error: {response['error_message']}\n{response['stack_trace']}")

                time.sleep(1e-4)

    def run_test(self, x):
        if self.online_flag:
            return self.run_command("server_test_func", args=[x])

    @contextmanager
    def locked(self):
        self.lock()
        try:
            yield
        finally:
            self.unlock()