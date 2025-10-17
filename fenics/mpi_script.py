from mpi4py import MPI

from fenics.local_data_manipulation.yaml_maker import get_node_data, get_output_dir
from fenics.simulator_mpi import Simulator_MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

node_id = rank

output_dir = get_output_dir()
node_type, neighbors, dataset_path = get_node_data(node_id, output_dir)

node_simulator = Simulator_MPI(
    node_id=node_id,
    node_dataset_path=dataset_path,
    type=node_type,
    neighbors=neighbors
    )

node_simulator.get_own_info()

"""
Fake simulation for testing metrics
"""
node_simulator.run_simulation()

node_simulator.make_local_metrics()
    

