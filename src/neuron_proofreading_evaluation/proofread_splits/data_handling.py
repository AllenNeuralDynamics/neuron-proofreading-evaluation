"""
Created on Wed Apr 8 14:00:00 2026

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Code for loading data to evaluate split correction pipeline.

"""

from segmentation_skeleton_metrics.data_handling.graph_loading import (
    GraphLoader,
    LabelHandler,
)
from segmentation_skeleton_metrics.utils import util
from segmentation_skeleton_metrics.utils.img_util import TensorStoreImage

import numpy as np
import pandas as pd


# --- Data Loading ---
def load_groundtruth(
    segmentation_path,
    swcs_path,
    anisotropy=(0.748, 0.748, 1.0),
    label_handler=None,
):
    print("\nStep 1: Load Ground Truth")
    graph_loader = GraphLoader(
        anisotropy=anisotropy,
        is_groundtruth=True,
        label_handler=label_handler,
        segmentation=TensorStoreImage(img_path=segmentation_path),
        use_anisotropy=True,
    )
    return graph_loader(swcs_path)


def load_fragments(
    swcs_path,
    label_handler,
    anisotropy=(0.748, 0.748, 1.0),
    swc_names=set(),
    use_anisotropy=False,
):
    graph_loader = GraphLoader(
        anisotropy=anisotropy,
        is_groundtruth=False,
        label_handler=label_handler,
        swc_names=swc_names,
        use_anisotropy=use_anisotropy,
    )
    graphs = graph_loader(swcs_path)
    return graphs


def load_label_pairs(path):
    label_pairs = list()
    for label_pair_str in util.read_txt(path).splitlines():
        id1, id2 = sorted(label_pair_str.replace(" ", "").split(","))
        label_pairs.append((id1, id2))
    return label_pairs


def load_proposals_df(path, proposal_type=None, threshold=0):
    # Read and reformat csv
    proposals_df = pd.read_csv(path)
    proposals_df["Proposal"] = proposals_df["Proposal"].apply(clean_tuple)
    proposals_df = proposals_df.set_index("Proposal")

    # Extract sub df
    if proposal_type == "leaf2leaf":
        proposals_df = proposals_df[proposals_df["Leaf2Leaf"]]
    elif proposal_type == "leaf2branch":
        proposals_df = proposals_df[~proposals_df["Leaf2Leaf"]]
    return proposals_df[proposals_df["Prediction"] > threshold]


# --- Graph Operations ---
def apply_segment_labeling(graphs, labels):
    segment_ids = [util.get_segment_id(lbl) for lbl in labels]
    label_handler = LabelHandler(labels=segment_ids)
    for key, graph in graphs.items():
        graph.relabel_nodes(label_handler)


def build_graphs_at_threshold(
    t, gt_graphs, fragment_graphs, labels, proposals_df
):
    # Label handler
    proposals_df_t = proposals_df[proposals_df["Prediction"] > t]
    label_pairs = list(proposals_df_t.index)
    label_handler = LabelHandler(labels=labels, label_pairs=label_pairs)

    # Build fragment graphs
    fragment_graphs = (
        update_and_merge_graphs(fragment_graphs, label_handler, proposals_df_t)
        if t > 0.19
        else None
    )

    # Build ground truth graphs
    for graph in gt_graphs.values():
        graph.relabel_nodes(label_handler)

    return gt_graphs, fragment_graphs


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
    new_graphs = dict()
    node2name = dict()
    for key, graph in graphs.items():
        class_id = label_handler.get(key)
        if class_id not in new_graphs:
            new_graphs[class_id] = graph
            node2name[class_id] = [key] * graph.number_of_nodes()
        else:
            new_graphs[class_id].add_graph(graph, set_kdtree=False)
            node2name[class_id].extend([key] * graph.number_of_nodes())
    set_kdtrees(graphs)
    return new_graphs, node2name


def merge_proposals(graphs, label_handler, proposals_df):
    proposals_df = proposals_df.reset_index(drop=True).copy()
    for i in proposals_df.index:
        # Extract proposal info
        id1 = str(proposals_df["Segment1"][i])
        class_id1 = label_handler.get(id1)

        # Connect fragments
        if class_id1 in graphs:
            xyz1 = parse_coord_str(proposals_df["World1"][i])
            xyz2 = parse_coord_str(proposals_df["World2"][i])

            d1, node1 = graphs[class_id1].kdtree.query(xyz1)
            d2, node2 = graphs[class_id1].kdtree.query(xyz2)
            if d1 < 10 and d2 < 10:
                graphs[class_id1].add_highlighted_edge(node1, node2)


def relabel_nodes_wrt_graph(gt_graphs, fragment_graphs):
    # Create segment graphs
    labels = list(fragment_graphs.keys())
    label_handler = LabelHandler(labels=labels, use_segment_mapping=True)
    segment_graphs, node2label = combine_graphs(fragment_graphs, label_handler)

    # Relabel ground truth graphs
    for gt_graph in gt_graphs.values():
        node_label = ["0"] * gt_graph.number_of_nodes()
        for i in [i for i in gt_graph.nodes if gt_graph.node_label[i] != "0"]:
            # Node info
            segment_id = gt_graph.node_label[i]
            xyz = gt_graph.node_xyz(i)

            # Find closest fragment node
            if segment_id in segment_graphs:
                dist, node = segment_graphs[segment_id].kdtree.query(xyz)
                if dist < 20:
                    node_label[i] = node2label[segment_id][node]

        gt_graph.node_label = np.array(node_label)
        gt_graph.fix_label_misalignments()


def update_and_merge_graphs(graphs, label_handler, proposals_df):
    """
    Applies label updates and merge proposals into the graph collection.
    """
    # Update fragment graph label
    for graph in graphs.values():
        graph.update_label(label_handler)

    # Combine graphs and add proposals as edges
    graphs, _ = combine_graphs(graphs, label_handler)
    merge_proposals(graphs, label_handler, proposals_df)
    return graphs


# --- Helpers ---
def clean_tuple(t):
    """
    Normalizes a tuple-like string into a standardized ordered tuple.

    Parameters
    ----------
    t : str
        Input representing a tuple, typically a string like "('a', 'b')".

    Returns
    -------
    Tuple[str]
        A tuple containing two cleaned identifiers, sorted lexicographically.
    """
    proposal = str(t).translate(str.maketrans("", "", "()'"))
    id1, id2 = sorted(proposal.replace(" ", "").split(","))
    return (id1, id2)


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
    return graphs


def parse_coord_str(s):
    """
    Parses a space-separated coordinate string into a NumPy array.

    Parameters
    ----------
    s : str
        String representing a coordinate list.

    Returns
    -------
    numpy.ndarray
        1D array of floats parsed from the input string.
    """
    return np.fromstring(s.strip("[]"), sep=" ")


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
