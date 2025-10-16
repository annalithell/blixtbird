# fenics/node/attacks/freerider.py

import torch
import logging
from mpi4py import MPI
import pickle
from typing import Optional
from fenics.node.nodetype import NodeType

from fenics.node.base import BaseNode


class Node(BaseNode):
    """ Free-rider attack that intercepts model parameters without participating in training. """
    
    def __init__(self, node_id: int,neighbors: [int], data_path: str, logger: Optional[logging.Logger] = None):
        """
        Initialize a standard node
        
        Args:
            node_id: ID of the attacker node
            logger: Logger instance
        """
        super().__init__(node_id,neighbors, data_path, logger)
        self.type = NodeType.NORMAL
        self.comm = MPI.COMM_WORLD
        #Hard to learn an old dog new tricks
        self.neighbor_models = {}
    def train_model(self):
        """
        Standard training of model. 

        Returns:
            Model parameters of the node
        
        """


        return
        
    def execute(self):
        """
        Execution function:
            - Calls the train_model() function for a standard node

        """
        self.model_params = self.train_model()

        #self.logger.info(f"[node_{self.node_id}] is a normal node and do nothing")
        #return model.parameters()

    def aggregate(self):
        pass
    def send(self):
        send_data = pickle.dumps(self.model_params,protocol=-1)
        for i in self.neighbors:
            #implement protocols here, if you dont want to send data to every node just send them an empty message.
            self.comm.Isend(send_data,i,0)
            
    def recv(self):
        status = MPI.Status()
        #Keeps track of all recieve requests
        recv_requests = []
        #Creates a buffer to store the models before deseriliazing
        recv_buffer = {}
        for i in self.neighbors:
            #Probe the incoming message for size
            self.comm.Probe(source=i,tag=0,status=status)
            count = status.Get_count(MPI.BYTE)
            #Create a buffer with the right size
            recv_buffer[i] = bytearray(count)

            #Create a reqeuest and append it to the waiting list
            req = self.comm.Irecv([recv_buffer[i],count,MPI.BYTE],source=i,tag=0)
            recv_requests.append(req)
            
        #Wait for all transfers to complete this is what synchronizes the entire system
        MPI.Request.waitall(recv_requests)

        #convert the bytestreams to statedict
        for i in self.neighbors:
            try:
                self.neighbor_models[i] = pickle.loads(bytes(recv_buffer[i]))
                if not(len(self.neighbor_models[i])):
                    self.neighbor_models[i] = None
            except:
                print("Error unpickling model from neigbhor {i}")
                self.neighbor_models[i] = None

        """
        #This is exeperimental as fuck no clue what im doing
        #This is dumb because it only recieves them in order dont think mpi discards them but still dumb
        #However this ensure that you get a message from every node and thus that they are in sync
            while True:
              msg_waiting = self.comm.Iprobe(source=i,tag=MPI.ANY_TAG,status=status)
              if msg_waiting:
                neightbor_model = pickle.loads(self.comm.recv(source=i,tag=0))
                if(len(neightbor_model)<= 1):
                    self.neighbor_models[i] = None
                self.neighbor_models[i] = neightbor_model
                break
            """

        pass
