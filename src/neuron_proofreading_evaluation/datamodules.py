"""
Created on Wed Apr 8 14:00:00 2026

@author: Anna Grim
@email: anna.grim@alleninstitute.org

Code for loading data to evaluate split correction pipeline.

"""

from collections import defaultdict
from copy import deepcopy
from segmentation_skeleton_metrics.datamodules.graph_loading import (
    GraphLoader,
    LabelHandler,
)
from segmentation_skeleton_metrics.utils import util
from segmentation_skeleton_metrics.utils.img_util import TensorStoreImage
from tqdm import tqdm

import numpy as np
import os
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
    anisotropy=(0.748, 0.748, 1.0),
    label_handler=None,
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


def load_labels(path):
    return set(util.read_txt(path).splitlines())


def load_merge_predictions(csv_path):
    df = pd.read_csv(csv_path)
    df["xyz"] = df["xyz"].apply(
        lambda s: tuple(float(x) for x in re.findall(r"np\.float\d+\(([-\d.eE+]+)\)", s))
    )
    return df


def load_multiround_proposal_df(csv_paths):
    def get_value(idx):
        name, _ = os.path.splitext(csv_path)
        part = os.path.basename(name).split("_")[idx]
        return float(part.split("=")[-1])

    df_list = list()
    last_round_id = 0
    for csv_path in csv_paths:
        # Extract info
        only_leaf2leaf = "leaf2leaf" in csv_path
        round_id = get_value(-2)
        threshold = get_value(-1)

        # Collect results
        assert round_id > last_round_id
        df_list.append(load_proposal_df(csv_path, only_leaf2leaf, threshold))
        last_round_id = round_id
    return df_list


def load_proposal_df(csv_path, only_leaf2leaf=False, threshold=0):
    df = pd.read_csv(csv_path).reset_index(drop=True)
    df["Prediction"] = df["Prediction"].apply(float)
    df["Proposal"] = df["Proposal"].apply(clean_tuple)
    return get_subdf(df, only_leaf2leaf, threshold)


# --- Graph Operations ---
def apply_split_corrections(gt_graphs, fragment_graphs, labels, proposals_df):
    # Label handler
    label_pairs = proposals_df["Proposal"].tolist()
    label_handler = LabelHandler(labels=labels, label_pairs=label_pairs)

    # Build fragment graphs
    fragment_graphs = update_and_merge_graphs(
        fragment_graphs, label_handler, proposals_df
    )

    # Build ground truth graphs
    for graph in gt_graphs.values():
        graph.relabel_nodes(label_handler)
        graph.fix_label_misalignments()

    # Relabel fragments
    for graph in fragment_graphs.values():
        graph.label = label_handler.get(graph.name)

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
    # Group graphs by class_id
    groups = defaultdict(list)
    for key, graph in graphs.items():
        groups[label_handler.get(key)].append((key, graph))

    # Merge each group once
    new_graphs = dict()
    node2name = dict()
    for class_id, members in groups.items():
        key0, graph0 = members[0]
        new_graphs[class_id] = deepcopy(graph0)
        node2name[class_id] = [key0] * graph0.number_of_nodes()
        for key, graph in members[1:]:
            new_graphs[class_id].add_graph(graph, set_kdtree=False)
            node2name[class_id].extend([key] * graph.number_of_nodes())

    set_kdtrees(new_graphs)
    return new_graphs, node2name


def drop_filtered_labels(graphs, labels):
    segment_ids = [util.get_segment_id(u) for u in labels]
    label_handler = LabelHandler(labels=segment_ids)
    for key, graph in graphs.items():
        graph.relabel_nodes(label_handler)


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


def relabel_fragments_with_name(fragment_graphs):
    for graph in fragment_graphs.values():
        graph.label = graph.name


def relabel_groundtruth_wrt_fragments(gt_graphs, fragment_graphs):
    segment_graphs, node2label = _build_segment_graphs(fragment_graphs)
    for gt_graph in gt_graphs.values():
        _relabel_gt_graph(gt_graph, segment_graphs, node2label)


def update_and_merge_graphs(fragment_graphs, label_handler, proposals_df):
    """
    Applies label updates and merge proposals into the graph collection.
    """
    fragment_graphs, _ = combine_graphs(fragment_graphs, label_handler)
    merge_proposals(fragment_graphs, label_handler, proposals_df)
    return fragment_graphs


def _build_segment_graphs(fragment_graphs):
    # Group fragment graphs by segment_id
    segment_to_ccs = defaultdict(list)
    for key, graph in fragment_graphs.items():
        segment_to_ccs[key.split(".")[0]].append((key, graph))

    # Merge CCs per segment into a single graph for KD-tree queries
    segment_graphs = dict()
    node2label = dict()
    iterator = tqdm(segment_to_ccs.items(), desc="Build Segment Graphs")
    for segment_id, members in iterator:
        # Create new fragment graph
        key0, graph0 = members[0]
        combined = deepcopy(graph0)

        # Add fragment graphs with same segment ID
        labels = [key0] * combined.number_of_nodes()
        for key, graph in members[1:]:
            combined.add_graph(graph, set_kdtree=False)
            labels.extend([key] * graph.number_of_nodes())

        # Set graph info
        combined.set_kdtree()
        segment_graphs[segment_id] = combined
        node2label[segment_id] = labels
    return segment_graphs, node2label


def _relabel_gt_graph(gt_graph, segment_graphs, node2label):
    node_label = ["0"] * gt_graph.number_of_nodes()
    for i in gt_graph.nodes:
        # Check for null label
        if gt_graph.node_label[i] == "0":
            continue

        # Get segment ID of node
        segment_id = str(gt_graph.node_label[i])
        if segment_id not in segment_graphs:
            continue

        # Update label to closest fragment
        xyz = gt_graph.node_xyz(i)
        dist, node = segment_graphs[segment_id].kdtree.query(xyz)
        if dist < 20:
            node_label[i] = node2label[segment_id][node]

    gt_graph.node_label = np.array(node_label)
    gt_graph.fix_label_misalignments()


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
    for graph in graphs.values():
        graph.node_voxel[:, [0, 2]] = graph.node_voxel[:, [2, 0]]
    return graphs


def get_subdf(df, only_leaf2leaf, threshold):
    if only_leaf2leaf:
        df = df[df["Leaf2Leaf"]]
    return df[df["Prediction"] > threshold]


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


def set_graph_color(graphs, color):
    for graph in graphs.values():
        graph.set_color(color)


def set_kdtrees(graphs):
    """
    Sets "kdtree" attribute for a collection of graphs.

    Parameters
    ----------
    graph : Dict[str, FragmentGraph]
        Graphs to be updated.
    """
    for graph in graphs.values():
        graph.set_kdtree()
