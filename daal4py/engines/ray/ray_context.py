import ray
from daal4py.engines.context import Context

class RayContext(Context):
            
    def current_node_id(self):
        return ray.services.get_node_ip_address()

    def node_ids(self):
        return ray.state.node_ids()

    def available_resources(self):
        return ray.available_resources()

    def get_world_size(self):
        return len(ray.nodes())

    def cluster_resources(self):
        return ray.cluster_resources()


