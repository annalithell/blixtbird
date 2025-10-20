# fenics/node/attacks/freerider.py

import torch
import torch.nn as nn
import logging
from mpi4py import MPI
import pickle
from typing import Optional
from fenics.node.node_type import NodeType
from fenics.models.cnn import Net
import time

from fenics.node.abstract import AbstractNode

class NormalNode(AbstractNode):
    """ Free-rider attack that intercepts model parameters without participating in training. """
    
    def __init__(self, node_id: int, neighbors: Optional[int], data_path: str, logger: Optional[logging.Logger] = None):
        """
        Initialize a standard node
        
        Args:
            node_id: ID of the attacker node
            logger: Logger instance
        """
        super().__init__(node_id, neighbors, data_path, logger)
        self.node_type = NodeType.NORMAL
        self.comm = MPI.COMM_WORLD
        #Hard to learn an old dog new tricks
        #Map form node ids to neigbouring models
        self.neighbor_models = {}
        #a map of from ndoe ids to data sizes for you and your neighbors
        self.model = Net() # TODO change s.t. this is determined by user-defined config.yaml
        self.data_sizes = {}
        self.data_sizes[self.node_id] = 0 #insert len of data here


    def train_model(self, train_dataset, epochs=5):
        """
        Standard training of model. 

        Returns:
            Model parameters of the node
        
        """
        model = self.model
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.NLLLoss()
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

        start_time = time.time()
        model.train()
        self.logger.info(f"[Node {self.node_id}] Training for {epochs} epochs...")

        for epoch in range(epochs):
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

            self.logger.info(f"[Node {self.node_id}] Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

        training_time = time.time() - start_time
        return model.state_dict(), training_time
        
    def execute(self, epochs=5):
        """
        Execution function:
            - Calls the train_model() function for a standard node

        """
        train_dataset = torch.load(self.data_path, weights_only=False)
        self.model_params, self.training_time = self.train_model(train_dataset, epochs)
        self.logger.info(f"[Node {self.node_id}] Training finished in {self.training_time:.2f}s")
        
        #self.logger.info(f"[node_{self.node_id}] is a normal node and do nothing")
        #return model.parameters()

    def aggregate(self):
        """
        Saftey check, probably not needed
        if not models_state_dicts:
            self.logger.warning("No models to aggregate.")
            return None
        """
        # Initialize an empty state dict for the aggregated model
        aggregated_state_dict = {}
        total_data = sum(self.data_sizes.values())
        
        # Get the list of all parameter keys
        #param_keys = list(models_state_dicts[0].keys())
        param_keys = list(self.model_params.keys())
        
        for key in param_keys:
            # Initialize a tensor for the weighted sum
            #weighted_sum = torch.zeros_like(models_state_dicts[0][key])
            weighted_sum = torch.zeros_like(self.model_params[key])
            #add your own weights first
            weighted_sum+=self.model_params[key] * self.data_sizes[self.node_id]
            #for state_dict, size in zip(models_state_dicts, data_sizes):
            #add your neighbours weights
            for model_id in self.neighbor_models:
                weighted_sum += self.neighbor_models[model_id][key] * self.data_sizes[model_id]
            #for neighbor_model,size in zip(self.neighbor_models,self.data_sizes):
                #weighted_sum += neighbor_model[key] * size
            # Compute the weighted average
            self.model_params[key] = weighted_sum / total_data
        

    def send(self):
        send_data = pickle.dumps(self.model_params,protocol=-1)
        for i in self.neighbors:
            #implement protocols here, if you dont want to send data to every node just send them an empty message and 0 in data length.
            self.comm.Isend(send_data, i, 0)
            self.comm.Isend(self.data_sizes[i],i,1)
            
    def recv(self):
        status = MPI.Status()
        #Keeps track of all recieve requests
        recv_requests = []
        #Creates a buffer to store the models before deseriliazing
        recv_buffer_model = {}
        recv_buffer_datalen = {}
        for i in self.neighbors:
            #Probe the incoming message for size
            self.comm.Probe(source=i, tag=0, status=status)
            count = status.Get_count(MPI.BYTE)
            #Create a buffer with the right size
            recv_buffer_model[i] = bytearray(count)

            #Create a reqeuest and append it to the waiting list
            req = self.comm.Irecv([recv_buffer_model[i],count,MPI.BYTE],source=i,tag=0)
            req2 = self.comm.Irecv([recv_buffer_datalen[i],1,MPI.INT],source=i,tag=1)
            recv_requests.append(req)
            recv_requests.append(req2)
            
        #Wait for all transfers to complete this is what synchronizes the entire system
        MPI.Request.waitall(recv_requests)

        #convert the bytestreams to statedict
        for i in self.neighbors:
            try:
                self.neighbor_models[i] = pickle.loads(bytes(recv_buffer_model[i]))
                self.data_sizes[i] = recv_buffer_datalen[i]
                if (self.data_sizes[i] == 0):
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
