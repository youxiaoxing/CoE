class Graph:
    def __init__(self):
        self.nodes = set() 
        self.edges = {}
        self.edge_props = {}
    
    def add_node(self, node):
        if node not in self.nodes:
            self.nodes.add(node)
            self.edges[node] = set()
            return True
        return False
    
    def isEmpty(self):
        return len(self.nodes) == 0
    
    def empty(self):
        self.nodes = set()
        self.edges = {}
        self.edge_props = {}
        
    def add_edge(self, node1, node2, relation=""):
        self.add_node(node1)
        self.add_node(node2)
        
        edge = (node1, node2)
        
        if node2 not in self.edges[node1]:
            self.edges[node1].add(node2)
            self.edge_props[edge] = relation
            return True
        elif edge in self.edge_props and self.edge_props[edge] != relation and relation:
            self.edge_props[edge] = relation
            return True
        return False
    
    def remove_node(self, node):
        if node in self.nodes:
            for neighbor in self.edges[node]:
                self.edges[neighbor].remove(node)
                edge = (node, neighbor) if node <= neighbor else (neighbor, node)
                if edge in self.edge_props:
                    del self.edge_props[edge]
            
            del self.edges[node]
            self.nodes.remove(node)
            return True
        return False
    
    def remove_edge(self, node1, node2):
        if node1 in self.nodes and node2 in self.nodes:
            if node2 in self.edges[node1]:
                self.edges[node1].remove(node2)
                self.edges[node2].remove(node1)
                
                edge = (node1, node2) if node1 <= node2 else (node2, node1)
                if edge in self.edge_props:
                    del self.edge_props[edge]
                return True
        return False
    
    def get_neighbors(self, node):
        if node in self.nodes:
            return self.edges[node]
        return set()
    
    def get_edge_relation(self, node1, node2):
        edge = (node1, node2) if node1 <= node2 else (node2, node1)
        return self.edge_props.get(edge, "")
    
    def get_graph_string(self):
        result = []
        result.append("Graph structure:")
        result.append("Node: " + str(self.nodes))
        result.append("Edge:")
        for node in self.nodes:
            for neighbor in self.edges[node]:
                if node <= neighbor:
                    relation = self.get_edge_relation(node, neighbor)
                    relation_str = f" [{relation}]" if relation else ""
                    result.append(f"  {node} -> {neighbor}{relation_str}")
        
        return "\n".join(result)