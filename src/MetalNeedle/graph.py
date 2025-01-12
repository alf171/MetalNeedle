import matplotlib.pyplot as plt
import networkx as nx

class ComputationGraph:
    def __init__(self):
        self.graph = nx.DiGraph()

    def add_node(self, name, **attrs):
        """Add a node to the computation graph."""
        self.graph.add_node(name, **attrs)

    def add_edge(self, from_node, to_node):
        """Add a directed edge between two nodes."""
        self.graph.add_edge(from_node, to_node)

    def visualize(self, save_path=None):
        """Visualize the computation graph."""
        pos = nx.spring_layout(self.graph)  # Layout algorithm for positioning nodes
        node_labels = nx.get_node_attributes(self.graph, "label")

        # Draw nodes
        nx.draw_networkx_nodes(self.graph, pos, node_size=700, node_color="lightblue")
        # Draw edges
        nx.draw_networkx_edges(self.graph, pos, arrows=True, arrowstyle="->")
        # Draw labels
        nx.draw_networkx_labels(self.graph, pos, font_size=12, font_color="black")

        # Add optional node-specific labels
        if node_labels:
            nx.draw_networkx_labels(self.graph, pos, labels=node_labels, font_size=10)

        # Save or show the plot
        if save_path:
            plt.savefig(save_path, format="png")
        plt.show()