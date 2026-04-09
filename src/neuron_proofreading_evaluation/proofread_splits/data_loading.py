"""
Created on Wed Apr 8 14:00:00 2026

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Code for loading data to evaluate split correction pipeline.

"""

from segmentation_skeleton_metrics.data_handling.graph_loading import (
    GraphLoader
)
from segmentation_skeleton_metrics.utils import util
from segmentation_skeleton_metrics.utils.img_util import TensorStoreImage
from tqdm import tqdm

import networkx as nx
import pandas as pd


def load_groundtruth(
    segmentation_path,
    swcs_path,
    label_handler,
    anisotropy=(0.748, 0.748, 1.0),
):
    print("\nStep 1: Load Ground Truth")
    graph_loader = GraphLoader(
        anisotropy=anisotropy,
        is_groundtruth=True,
        label_handler=label_handler,
        label_mask=TensorStoreImage(img_path=segmentation_path),
        use_anisotropy=True,
    )
    return graph_loader(swcs_path)


def load_fragments(swcs_path, label_handler, anisotropy=(0.748, 0.748, 1.0)):
    print("\nStep 2: Load Fragments")
    graph_loader = GraphLoader(
        anisotropy=anisotropy,
        is_groundtruth=False,
        label_handler=label_handler,
        selected_ids=label_handler.valid_labels,
        use_anisotropy=False,
    )
    graphs = graph_loader(swcs_path)
    flip_coordinates(graphs)
    return graphs


def load_corrected_fragments(
    swcs_path, label_handler, anisotropy=(0.748, 0.748, 1.0)
):
    print("\nStep 2: Load Fragments")
    graph_loader = GraphLoader(
        anisotropy=anisotropy,
        is_groundtruth=False,
        label_handler=label_handler,
        selected_ids=label_handler.valid_labels,
        use_anisotropy=True,
    )
    return graph_loader(swcs_path)


def load_precorrected_fragments(
    swcs_path, label_handler, anisotropy=(0.748, 0.748, 1.0)
):
    print("\nStep 2: Load Fragments")
    graph_loader = GraphLoader(
        anisotropy=anisotropy,
        is_groundtruth=False,
        label_handler=label_handler,
        selected_ids=label_handler.valid_labels,
        use_anisotropy=True,
    )
    return graph_loader(swcs_path)


def load_proposals_df(path, proposal_type):
    proposals_df = pd.read_csv(path).set_index("Proposal")
    if proposal_type == "leaf2leaf":
        idxs = proposals_df["Leaf2Leaf"]
    elif proposal_type == "leaf2branch":
        idxs = ~proposals_df["Leaf2Leaf"]
    else:
        return proposals_df
    return proposals_df[idxs]


# --- Label Handler ---
class CorrectedLabelHandler:
    """
    Handles mapping between raw segmentation labels and consolidated class IDs.

    The class is designed to manage cases where multiple segment IDs are merged
    into a single equivalence class. It supports:
      - Building mappings from a file of pairwise segment connections.
      - Mapping individual labels to class IDs.
      - Retrieving all labels belonging to a given class.
      - Enforcing constraints on which labels are considered valid.

    Attributes
    ----------
    mapping : Dict[int, int]
        Maps a raw label (segment ID) to its class ID.
    inverse_mapping : Dict[int, Set[int]]
        Maps a class ID back to the set of raw labels it contains.
    valid_labels : Set[int]
        Labels that are allowed to be assigned (after filtering).
    """

    def __init__(self, label_pairs, labels_path):
        """
        Instantiates a LabelHandler object and optionally builds label
        mappings.

        Parameters
        ----------
        label_pairs : List[Tuple[str]]
            Pairs of SWC IDs that were merged.
        valid_labels : Set[int]
            Subset of labels that are considered to be valid. This argument
            accounts for segments removed due to filtering.
        """
        
        self.mapping = dict()  # Maps label to equivalent class id
        self.inverse_mapping = dict()  # Maps class id to list of labels
        self.valid_labels = set(util.read_txt(labels_path).splitlines())
        self.init_mappings(label_pairs)

        print("133 - Check:", "1085129783.0" in self.valid_labels)

    # --- Constructor Helpers ---
    def init_mappings(self, label_pairs):
        """
        Initializes dictionaries that map between segment IDs and equivalent
        class IDS.

        Parameters
        ----------
        label_pairs : str
            Pairs of SWC IDs that were merged.
        """
        self.mapping = {0: 0}
        self.inverse_mapping = {0: [0]}
        labels_graph = self.build_labels_graph(label_pairs)
        for i, labels in enumerate(nx.connected_components(labels_graph)):
            class_id = i + 1
            self.inverse_mapping[class_id] = set()
            for label in labels:
                self.mapping[label] = class_id
                self.inverse_mapping[class_id].add(label)

    def build_labels_graph(self, label_pairs):
        """
        Builds a graph of labels from valid labels and merge connections.
        Nodes correspond to "self.valid_labels", and edges are added between
        labels that were merged according to the file.

        Parameters
        ----------
        label_pairs : str
            Pairs of SWC IDs that were merged.

        Returns
        -------
        labels_graph : networkx.Graph
            Graph with nodes that represent labels and edges are based on the
            connections read from the "connections_path".
        """
        labels_graph = nx.Graph()
        labels_graph.add_nodes_from(self.valid_labels)
        labels_graph.add_edges_from(label_pairs)
        print("176 - Check:", "1085129783.0" in labels_graph.nodes)
        return labels_graph

    # --- Core Routines ---
    def get(self, label):
        """
        Maps a raw label to its class ID.

        Parameters
        ----------
        label : int
            Raw label (segment ID) to be mapped.

        Returns
        -------
        int
            Class ID corresponding to the label.
        """
        return 0 if label not in self.valid_labels else self.mapping[label]

    # --- Helpers ---
    def node_labels(self, graph):
        """
        Gets the set of unique node labels from the given graph.

        Parameters
        ----------
        graph : LabeledGraph
            Graph from which to retrieve the node labels.

        Returns
        -------
        labels : Set[int]
            Labels corresponding to nodes in the graph identified by "key".
        """
        return set().union(*(self.inverse_mapping[u] for u in labels))


# --- Helpers ---
def combine_graphs(graphs, label_handler):
    """
    Combines graphs with the same label.

    Parameters
    ----------
    graph : Dict[str, FragmentGraph]
        Graphs to be updated.

    Returns
    -------
    new_graphs : Dict[str, FragmentGraph]
        Updated graphs.
    """
    print("229 - Checking:", "1085129783.0" in graphs.keys())
    new_graphs = dict()
    for key, graph in tqdm(graphs.items(), desc="Combine Graphs"):
        class_id = label_handler.get(key)
        if class_id not in new_graphs:
            new_graphs[class_id] = graph
        else:
            new_graphs[class_id].add_graph(graph, set_kdtree=False)
    set_kdtrees(graphs)
    print("# Graphs:", len(graphs))
    print("# New Graphs:", len(new_graphs))
    return new_graphs


def flip_coordinates(graphs):
    """
    Flips the X and Z coordinates for a collections of graphs.

    Parameters
    ----------
    graph : Dict[str, FragmentGraph]
        Graphs to be updated.
    """
    for key, graph in graphs.items():
        graphs[key].node_voxel[:, [0, 2]] = graph.node_voxel[:, [2, 0]]


def set_kdtrees(graphs):
    """
    Sets "kdtree" attribute for a collection of graphs.

    Parameters
    ----------
    graph : Dict[str, FragmentGraph]
        Graphs to be updated.
    """
    for key in graphs:
        graphs[key].set_kdtree()
