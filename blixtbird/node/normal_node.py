# blixtbird/node/normal_node.py

import torch
import torch.nn as nn
import logging
import pickle
from typing import Optional
import time
from mpi4py import MPI
from torchvision import datasets, transforms
import numpy as np

from blixtbird.node.node_type import NodeType
from blixtbird.node.abstract import AbstractNode
from blixtbird.training.evaluator import evaluate

class NormalNode(AbstractNode):
    """ Free-rider attack that intercepts model parameters without participating in training. """
    
    def __init__(self, node_id: int, data_path: str, neighbors: Optional[int], model_type, epochs, logger: Optional[logging.Logger] = None):
        """
        Initialize a standard node
        
        Args:
            node_id: ID of the attacker node
            logger: Logger instance
        """
        super().__init__(node_id, data_path, neighbors, model_type, epochs, logger)
        self.node_type = NodeType.NORMAL


    def train_model(self):
        """
        Standard training of model. 

        Returns:
            Model parameters of the node
        
        """
        device = torch.device("cpu")

        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3, weight_decay=1e-4)
        criterion = nn.NLLLoss()
        train_dataset = torch.load(self.data_path, weights_only=False)
        self.data_sizes[self.node_id] = len(train_dataset)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=32, shuffle=True)

        
        # Create test DataLoader
        transform = transforms.Compose([transforms.ToTensor()])
        test_dataset = datasets.FashionMNIST('./data', train=False, download=True, transform=transform)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

        start_time = time.time()
        self.model.train()

        print(f"[Node {self.node_id}] Training for {self.epochs} epochs...")

        for epoch in range(self.epochs):
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
            print(f"[Node {self.node_id}] Epoch {epoch+1}/{self.epochs}")

        # after round  evaluate train and test
        self.append_training_metrics(train_loader)
        self.append_test_metrics(test_loader)

        self.training_time = time.time() - start_time

    
    
    def append_training_metrics(self, train_loader):
        # Evaluation phase: training data
        train_loss, train_accuracy, train_f1, train_precision, train_recall = evaluate(self.model, train_loader)

        self.metrics_train.append({'train_loss': train_loss,
                                'train_accuracy': train_accuracy,
                                'train_f1_score': train_f1,
                                'train_precision': train_precision,
                                'train_recall':train_recall})
        

    def append_test_metrics(self, test_loader):
        # Evaluation phase: testing data
        loss, accuracy, f1, precision, recall = evaluate(self.model, test_loader)

        self.metrics_test.append({'test_loss': loss,
                                'test_accuracy': accuracy,
                                'test_f1_score': f1,
                                'test_precision': precision,
                                'test_recall': recall})

    def append_test_metrics_after_aggregation(self):
        #TODO Optimize this DataLoader code
        # Create test DataLoader
        transform = transforms.Compose([transforms.ToTensor()])
        test_dataset = datasets.FashionMNIST('./data', train=False, download=True, transform=transform)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=32, shuffle=False)

        # Evaluation phase: testing data
        loss, accuracy, f1, precision, recall = evaluate(self.model, test_loader)

        #aa stands for after agregation
        self.metrics_test_after_aggregation.append({'test_loss_aa': loss,
                                'test_accuracy_aa': accuracy,
                                'test_f1_score_aa': f1,
                                'test_precision_aa': precision,
                                'test_recall_aa': recall})

    def execute(self):
        """
        Execution function:
            - Calls the train_model() function for a standard node

        """
        # Train the model
        self.train_model()
        print(f"[Node {self.node_id}] Training finished in {self.training_time:.2f}s")
        # Send and receive model parameters
        print(f"[Node {self.node_id}] will now start sending data")
        self.send()
        print(f"[Node {self.node_id}] has completed sending, now stating the recieve operation")
        self.recv()
        # Perform aggregation of model parameters
        print(f"[Node {self.node_id}] Communication completed, starting aggregation .....")
        self.aggregate()
        # Evaluate model after aggregation
        self.append_test_metrics_after_aggregation()


    def aggregate(self):

        # Initialize an empty state dict for the aggregated model
        total_data = sum(self.data_sizes.values())
        
        # Get the list of all parameter keys
        param_keys = list(self.model.state_dict().keys())

        for key in param_keys:
            #Initialize empty sum to add
            weighted_sum = torch.zeros_like(self.model.state_dict()[key])
            #Add your own weights first
            weighted_sum+=self.model.state_dict()[key] * self.data_sizes[self.node_id]
            #Add you neighbours weights
            for model_id in self.neighbor_statedicts:
                #print(f"node {self.node_id} adds  {model_id} {key}")
                weighted_sum += self.neighbor_statedicts[model_id][key] * self.data_sizes[model_id]
           #print(f"{self.node_id} has weighted sum in key {key} of {weighted_sum} and total data of {total_data}")
            #Create a temp tensor to call pytorch library and set the tensor
            param_tensor = self.model.state_dict()[key]
            param_tensor.data.copy_(weighted_sum / total_data)
        
    def send(self):
        send_data = pickle.dumps(self.model.state_dict(),protocol=-1)
        #print(f"[Node {self.node_id}] has completed pickeling")
        send_size = len(send_data)
        for i in self.neighbors:    
            #implement protocols here, if you dont want to send data to every node just send them an empty message and 0 in data length.
            req1 = self.comm.Isend([send_data,send_size,MPI.BYTE], i, 0)
            #print(f"[Node {self.node_id}] has sent the pickled data to node {i}")
            #MPI needs cant send an obect so we have to wrap it in a an array because ahahhahahhahahah
            send_buffer = np.array([self.data_sizes[self.node_id]], dtype=np.int32)
            req2 = self.comm.Isend([send_buffer,1,MPI.INT],i,1)

            self.send_requests.append(req1)
            self.send_requests.append(req2)
            print(f"[Node {self.node_id}] has sent the datasize to node {i}")


            
    def recv(self):
        status = MPI.Status()
        #Keeps track of all recieve requests
        recv_requests = []
        #Creates a buffer to store the models before deseriliazing
        recv_buffer_model = {}
        recv_buffer_datalen = {}
        #print(f"[Node {self.node_id}] Has init the recieve")
        for i in self.neighbors:
            #Probe the incoming message for size
            self.comm.Probe(source=i, tag=0, status=status)
            count = status.Get_count(MPI.BYTE)
            #Create a buffer with the right size
            recv_buffer_model[i] = bytearray(count)
            #Create a reqeuest and append it to the waiting list
            #print(f"[Node {self.node_id}] has probed and created the recv buffer from {i}")
            req = self.comm.Irecv([recv_buffer_model[i],count,MPI.BYTE],source=i,tag=0)
            #print(f"[Node {self.node_id}] created the model buffer request for {i}")
            #initialize the recv buffer as a np array because MPI </3 Python
            recv_buffer_datalen[i] = np.array([0],dtype=np.int32)
            req2 = self.comm.Irecv([recv_buffer_datalen[i],1,MPI.INT],source=i,tag=1)
            #print(f"[Node {self.node_id}] has created the datasize buffer request for {i}")
            recv_requests.append(req)
            recv_requests.append(req2)
            #print(f"[Node {self.node_id}] has appended the requests")
            
        #Wait for all transfers to complete this is what synchronizes the entire system
        MPI.Request.waitall(self.send_requests + recv_requests)
        print(f"[Node {self.node_id}] has passed the barrier")

        #convert the bytestreams to statedict
        for i in self.neighbors:
            try:
                self.neighbor_statedicts[i] = pickle.loads(bytes(recv_buffer_model[i]))
                self.data_sizes[i] = recv_buffer_datalen[i][0]
                if (self.data_sizes[i] == 0):
                    self.neighbor_statedicts[i] = None
            except:
                print(f"Error unpickling model from neigbhor {i}")
            #    self.neighbor_statedicts[i] = None

        
