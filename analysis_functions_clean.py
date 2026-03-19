"""
analysis_functions.py
=====================
Image analysis pipeline for the FACCC (Fluid-Associated Cell-Cell Contact)
framework.  Computes tissue-scale mechanical and geometric parameters from
cell segmentation masks produced by Cellpose (or equivalent tools), including:

  - Cell-Cell Contact (CCC)
  - Fluid pocket statistics
  - Multi- and tri-cellular junction counts (MCJ / TCJ)
  - Contact angle alpha at fluid pockets
  - Rigidity network analysis (Giant Rigid Cluster, average connectivity)

Dependencies: numpy, pandas, scikit-image, scipy, networkx, matplotlib,
              tifffile, tqdm, pebble (lattice_BCM library).
"""

import os
import math
import json
import logging
import sys
import subprocess

import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl
import tifffile as tifi
import skimage.io as skio
from skimage import measure
from skimage.morphology import dilation, skeletonize
from scipy.stats import sem
from scipy.ndimage import generic_filter
from scipy import ndimage
from tqdm import tqdm
import pebble


logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(message)s',
    datefmt='%m-%d %T',
    stream=sys.stdout,
)


# ─────────────────────────────────────────────────────────────────────────────
# Geometry helpers
# ─────────────────────────────────────────────────────────────────────────────

def find_duplicate_arrays_with_indexes(lst):
    """Map each unique array in *lst* to all indices at which it appears.

    Uses the tuple hash of each array as the dictionary key, so the returned
    dict is keyed by those hashes (not by the arrays themselves).

    Returns
    -------
    dict : hash -> list of int indices
    """
    value_indexes = {}
    for index, array in enumerate(lst):
        key = hash(tuple(array))
        if key in value_indexes:
            value_indexes[key].append(index)
        else:
            value_indexes[key] = [index]
    return value_indexes


def find_median_coordinate(coordinates):
    """Return the element-wise median of a list of (row, col) tuples.

    Uses a sort-based median, taking the lower-middle element for even-length
    lists (matches numpy's default integer-median behaviour).
    """
    x_coords = [c[0] for c in coordinates]
    y_coords = [c[1] for c in coordinates]
    median_x = sorted(x_coords)[len(x_coords) // 2]
    median_y = sorted(y_coords)[len(y_coords) // 2]
    return (median_x, median_y)


def get_coordinates_in_radius(center, radius):
    """Return integer pixel positions that lie on a discrete circle.

    Both the horizontal and vertical sweeps are included so all perimeter
    pixels are covered even for small radii.

    Parameters
    ----------
    center : (int, int)   (row, col) of the circle centre.
    radius : int          Circle radius in pixels.

    Returns
    -------
    list of (int, int) tuples  (may contain duplicates near the axes).
    """
    x_center, y_center = center
    coordinates = []

    for x in range(x_center - radius, x_center + radius + 1):
        y_top    = math.floor(y_center + math.sqrt(radius**2 - (x - x_center)**2))
        y_bottom = math.floor(y_center - math.sqrt(radius**2 - (x - x_center)**2))
        coordinates.append((x, y_top))
        coordinates.append((x, y_bottom))

    for y in range(y_center - radius, y_center + radius + 1):
        x_right = math.floor(x_center + math.sqrt(radius**2 - (y - y_center)**2))
        x_left  = math.floor(x_center - math.sqrt(radius**2 - (y - y_center)**2))
        coordinates.append((x_right, y))
        coordinates.append((x_left,  y))

    return coordinates


def calculate_angle(a, b, c):
    """Compute the contact angle at vertex *b* along the path a – b – c.

    Parameters
    ----------
    a, b, c : (float, float)   (row, col) positions of the three points.

    Returns
    -------
    angle_degrees : float
        Interior angle at *b* in degrees (180° − the geometric angle between
        vectors BA and BC).
    cos_half_angle : float
        cos(angle_degrees / 2), used as the FACCC contact angle metric alpha.
    """
    ab = (b[0] - a[0], b[1] - a[1])
    bc = (c[0] - b[0], c[1] - b[1])
    dot_product  = ab[0] * bc[0] + ab[1] * bc[1]
    magnitude_ab = math.sqrt(ab[0]**2 + ab[1]**2)
    magnitude_bc = math.sqrt(bc[0]**2 + bc[1]**2)
    angle_degrees = 180 - math.degrees(math.acos(dot_product / (magnitude_ab * magnitude_bc)))
    cos_half      = math.cos(math.radians(angle_degrees) / 2)
    return angle_degrees, cos_half


def find_pixels_with_two_different_neighbors(image, neighbor_count):
    """Return (col, row) positions where exactly *neighbor_count* distinct
    label values appear among the 4-connected neighbours of a pixel.

    Used to locate precise boundary points within a small region patch during
    angle detection.
    """
    pixels = []
    for row in range(len(image)):
        for col in range(len(image[row])):
            current   = image[row][col]
            neighbors = []
            if row > 0                    and image[row - 1][col] != current:
                neighbors.append(image[row - 1][col])
            if row < len(image) - 1       and image[row + 1][col] != current:
                neighbors.append(image[row + 1][col])
            if col > 0                    and image[row][col - 1] != current:
                neighbors.append(image[row][col - 1])
            if col < len(image[row]) - 1  and image[row][col + 1] != current:
                neighbors.append(image[row][col + 1])
            if len(np.unique(neighbors)) == neighbor_count:
                pixels.append((col, row))
    return pixels


def average_close_points(coords, threshold_distance):
    """Merge nearby junction detections into single representative coordinates.

    Uses a greedy O(n²) pass: for each unassigned point the first point in the
    remaining list acts as the seed and all other points within
    *threshold_distance* are absorbed into it.  The cluster centroid is kept.

    Parameters
    ----------
    coords : np.ndarray, shape (N, 2)
        Raw junction pixel positions.
    threshold_distance : float
        Euclidean pixel distance below which two detections are merged.

    Returns
    -------
    np.ndarray of shape (M, 2), M ≤ N.
    """
    averaged = []
    used = np.zeros(len(coords), dtype=bool)
    for i in range(len(coords)):
        if not used[i]:
            cluster = [coords[i]]
            used[i] = True
            for j in range(i + 1, len(coords)):
                if not used[j] and np.linalg.norm(coords[i] - coords[j]) < threshold_distance:
                    cluster.append(coords[j])
                    used[j] = True
            averaged.append(np.mean(cluster, axis=0).astype(int))
    return np.array(averaged)


def count_unique_labels(neighborhood):
    """Count distinct non-zero cell labels in a pixel neighbourhood patch.

    Intended for use as the kernel function passed to `scipy.ndimage.generic_filter`.
    """
    unique = np.unique(neighborhood)
    return len(unique[unique != 0])


# ─────────────────────────────────────────────────────────────────────────────
# Pebble-game algorithm  (reference implementation, see also getClusters)
# ─────────────────────────────────────────────────────────────────────────────
# NOTE: these three functions are an independent Python implementation of the
# (2,3)-pebble game for 2D rigidity analysis.  The production pipeline uses
# the external ``pebble`` library instead (see getClusters).  They are
# retained here as a transparent reference.

def findPebble(v, _v, seen, path, pebble):
    """Depth-first search for a free pebble reachable from node *v*.

    Parameters
    ----------
    v      : int   Current node being explored.
    _v     : int   The bond partner of *v*; pebbles may not be borrowed from it.
    seen   : list of bool   Visited-node flags.
    path   : list of int    Traversal path (index of parent node, -1 at terminus).
    pebble : list of 2-tuples   Current pebble allocation per node.

    Returns True if a free pebble was found.
    """
    seen[v] = True
    path[v] = -1  # -1 signals the end of the augmenting path
    if pebble[v][0] is None or pebble[v][1] is None:
        return True
    for neighbour in (pebble[v][0], pebble[v][1]):
        if not seen[neighbour] and neighbour != _v:
            path[v] = neighbour
            if findPebble(neighbour, _v, seen, path, pebble):
                return True
    return False


def rearrangePebble(v, _v, path, pebble):
    """Reallocate pebbles along the path found by `findPebble`.

    After the call, node *v* holds a pebble that covers the bond (v, _v).
    Parameters are the same as in `findPebble`.
    """
    if path[v] == -1:
        # Node v already has a free slot; assign the new bond to it.
        if pebble[v][0] is None:
            pebble[v] = (_v, pebble[v][1])
        elif pebble[v][1] is None:
            pebble[v] = (pebble[v][0], _v)
        else:
            logger.error('rearrangePebble: unexpected state at terminus (loc 0).')
        return

    v_copy = v
    # Walk the augmenting path, shifting each pebble one step towards v.
    while path[v] != -1:
        w = path[v]
        if path[w] == -1:
            if pebble[w][0] is None:
                pebble[w] = (v, pebble[w][1])
            elif pebble[w][1] is None:
                pebble[w] = (pebble[w][0], v)
            else:
                logger.error('rearrangePebble: unexpected state at path end (loc 1).')
        else:
            _w = path[w]
            if pebble[w][0] == _w:
                pebble[w] = (v, pebble[w][1])
            elif pebble[w][1] == _w:
                pebble[w] = (pebble[w][0], v)
            else:
                logger.error('rearrangePebble: pebble not found on path (loc 2).')
        v = w

    # Update the origin node to point at the new bond.
    if pebble[v_copy][0] == path[v_copy]:
        pebble[v_copy] = (_v, pebble[v_copy][1])
    elif pebble[v_copy][1] == path[v_copy]:
        pebble[v_copy] = (pebble[v_copy][0], _v)
    else:
        logger.error('rearrangePebble: origin pebble not found (loc 3).')


def pebbleGame(G):
    """(2,3)-pebble game algorithm for identifying rigid sub-graphs.

    Each node holds at most 2 pebbles.  A bond is independent if 4 pebbles
    can be collectively gathered from both endpoints; otherwise it is
    redundant and its two endpoints belong to a rigid cluster.

    Parameters
    ----------
    G : list of adjacency lists   Each edge appears exactly once.

    Returns
    -------
    rigid_node : list of bool
        True for each node that belongs to a rigid component.
    pebble : list of 2-tuples
        Final pebble allocation.
    redundant : list of (int, int)
        Edges identified as redundant (over-constrained).
    """
    pebble_alloc = [(None, None)] * len(G)
    rigid_node   = [False] * len(G)
    redundant    = []

    for i, neighbours in enumerate(G):
        for v in neighbours:
            logger.info('Testing edge (%d, %d)', i, v)
            # Work on a copy to probe whether 4 pebbles can be gathered.
            pebble_cpy = json.loads(json.dumps(pebble_alloc))
            flag    = True
            seen    = [False] * len(G)
            seen_or = [False] * len(G)

            for attempt in range(4):
                path = [-1] * len(G)
                if findPebble(i, v, seen, path, pebble_cpy):
                    rearrangePebble(i, v, path, pebble_cpy)
                elif findPebble(v, i, seen, path, pebble_cpy):
                    rearrangePebble(v, i, path, pebble_cpy)
                else:
                    flag = False
                    if attempt < 3:
                        logger.error('pebbleGame: pebble search failed before attempt 3.')
                    else:
                        # Edge is redundant; nodes in seen_or form a rigid sub-graph.
                        redundant.append((i, v))
                        for x, is_seen in enumerate(seen_or):
                            if is_seen:
                                rigid_node[x] = True
                seen_or = [a or b for a, b in zip(seen, seen_or)]
                seen    = [False] * len(G)

            if flag:
                # Check the Laman condition: e = 2v − 3 implies rigidity.
                if sum(seen_or) != 2:
                    num_edge = sum(
                        1 for si, s in enumerate(seen_or) if s
                        for p in pebble_cpy[si]
                        if p is not None and seen_or[p]
                    )
                    if num_edge == 2 * sum(seen_or):
                        for x, is_seen in enumerate(seen_or):
                            if is_seen:
                                rigid_node[x] = True

                # Bond is independent; commit the pebble allocation.
                seen = [False] * len(G)
                path = [-1]   * len(G)
                if findPebble(i, v, seen, path, pebble_alloc):
                    rearrangePebble(i, v, path, pebble_alloc)
                elif findPebble(v, i, seen, path, pebble_alloc):
                    rearrangePebble(v, i, path, pebble_alloc)
                else:
                    logger.error('pebbleGame: could not place pebble for independent edge.')

    return rigid_node, pebble_alloc, redundant


# ─────────────────────────────────────────────────────────────────────────────
# Rigidity analysis via the external pebble library
# ─────────────────────────────────────────────────────────────────────────────

def getClusters(edges_df, N_cells_total=None):
    """Decompose the cell-contact network into rigid clusters using the
    (2,3)-pebble game from the ``pebble`` library.

    Parameters
    ----------
    edges_df : pd.DataFrame
        Must contain columns ``CellLabel`` and ``NeighboringCellLabel``
        (1-based cell indices, each directed edge listed once).
    N_cells_total : int, optional
        Total number of cells in the ROI, **including** isolated cells with
        no contacts.  Used to normalise avK, GRC, and GRC2.  When omitted,
        normalisation falls back to the number of nodes present in the
        contact network, which overestimates these quantities when isolated
        cells exist.

    Returns
    -------
    Giant_Cluster : list of (int, int)
        1-based (node_a, node_b) edge pairs in the Giant Rigid Cluster (GRC).
    Second_Giant_Cluster : list of (int, int)
        1-based edge pairs of the second-largest rigid cluster.
    Percolation_Cluster : list of (int, int)
        1-based edge pairs of all rigid clusters with more than 2 nodes.
    bernat_values : [int, float, float, float]
        [N, normalised_avK, GRC_fraction, GRC2_fraction].
    """
    sys.path.append('../Library/lattice_BCM')

    # Build a 0-based, deduplicated, sorted edge list.
    edges_raw = np.array(edges_df[['CellLabel', 'NeighboringCellLabel']]) - 1
    edges_sorted = [[min(a, b), max(a, b)] for a, b in edges_raw]
    file = np.unique(edges_sorted, axis=0)

    # Build the pebble-game lattice and register all bonds.
    G = pebble.lattice()
    for a, b in file:
        G.add_bond(int(a), int(b))
    G.stat()

    # Determine the normalisation denominator.
    # N_cells_total (all ROI cells) is preferred over N_network (bonded cells
    # only) to avoid overestimating connectivity and cluster fractions.
    N_network = G.statistics['site']
    N = N_cells_total if N_cells_total is not None else N_network

    # Normalised average contact number.
    # The reference maximum uses the triangular lattice formula for an N-cell tissue.
    Raw_avK           = 2 * G.statistics['bond'] / N
    Full_connectivity = (8 + 16 * (np.sqrt(N) - 2) + 6 * (np.sqrt(N) - 2)**2) / N
    Normalized_avK    = Raw_avK / Full_connectivity

    # Decompose the network into rigid clusters.
    G.decompose_into_cluster()

    size   = 0
    size_G = 0
    K      = 0
    K2     = 0
    Percolation_Cluster = []

    # First pass: find the Giant Rigid Cluster (GRC) and percolation cluster.
    for key, value in G.cluster['index'].items():
        n = len(np.unique(value))
        if n > 2:
            Percolation_Cluster.extend(value)
        if n > size:
            size = n
            K    = key

    GRC = size / N

    # Second pass: find the second-largest cluster.
    for key, value in G.cluster['index'].items():
        if key != K:
            n = len(np.unique(value))
            if n > size_G:
                size_G = n
                K2     = key

    GRC2 = size_G / N

    # Collect edge lists; K2 may be 0 (no second cluster) which can raise KeyError.
    Giant_Cluster        = list(G.cluster['index'][K])
    Second_Giant_Cluster = []
    try:
        Second_Giant_Cluster = list(G.cluster['index'][K2])
    except KeyError:
        pass

    # Convert 0-based node indices back to the 1-based convention used externally.
    Giant_Cluster1        = [(x + 1, y + 1) for x, y in Giant_Cluster]
    Second_Giant_Cluster1 = [(x + 1, y + 1) for x, y in Second_Giant_Cluster]
    PC1 = [(x + 1, y + 1) for x, y in np.array(Percolation_Cluster)]

    return Giant_Cluster1, Second_Giant_Cluster1, PC1, [N, Normalized_avK, GRC, GRC2]


# ─────────────────────────────────────────────────────────────────────────────
# Visualisation helpers
# ─────────────────────────────────────────────────────────────────────────────

def get_better_coordinates(touches, inters3, centers, str_arr, labeled_image, key,
                           plot, folderpath, averaged_coords,
                           ori=False, radiuspx=3, image_format='png'):
    """Save junction overlay figures for one image / bin.

    Two figures are written to ``<folderpath>/results/`` when *plot* is True:
      - ``<key>_tcj.<fmt>``    : segmented image with detected junction positions.
      - ``<key>_angles.<fmt>`` : image annotated with contact-angle values.

    Parameters
    ----------
    touches       : list of (row, col)   Raw junction pixel coordinates.
    inters3       : list of (row, col)   Angle-arm intersection points.
    centers       : list of (row, col)   Junction centre positions.
    str_arr       : list of str          Angle values as formatted strings.
    labeled_image : np.ndarray           Current cell-labelled segmentation.
    key           : str                  Sample name used for output file names.
    plot          : bool                 Whether to write figures to disk.
    folderpath    : str                  Base analysis directory.
    averaged_coords : np.ndarray         Deduplicated junction coordinates (N, 2).
    ori           : bool                 If True, load the raw TIFF for the angle overlay.
    radiuspx      : float                Deduplication radius in pixels.
    image_format  : str                  Output format ('svg' or 'png').
    """
    if not plot:
        return

    results_folder = os.path.join(folderpath, 'results', '')

    # Load the background image for the angle overlay.
    if ori:
        ori_name   = key.split('(RGB)')[0] if '(RGB)' in key else key
        ori_folder = '/'.join(folderpath.split('/')[:-4]) + '/ROI/' + '/'.join(folderpath.split('/')[-3:])
        ori_img    = tifi.imread(ori_folder + ori_name + '.tif')[1]
        if len(ori_img) == 3:
            ori_img = ori_img[0]
    else:
        ori_img = labeled_image

    # Junction position overlay on the segmentation.
    plt.figure()
    plt.imshow(labeled_image)
    if len(averaged_coords):
        plt.scatter(averaged_coords[:, 1], averaged_coords[:, 0], c='red', s=2)
    plt.axis('off')
    plt.savefig(results_folder + key + '_tcj.' + image_format, format=image_format, dpi=600)
    plt.close()

    # Contact-angle annotation overlay.
    xii, yii       = zip(*inters3)
    x_centers, y_centers = zip(*centers)
    plt.figure()
    plt.imshow(ori_img)
    plt.scatter(y_centers, x_centers, color='blue',   s=2)
    plt.scatter(yii,       xii,       color='yellow', s=1)
    for i, centre in enumerate(centers):
        plt.text(centre[1], centre[0], str_arr[i], fontsize=7)
    plt.axis('off')
    plt.savefig(results_folder + key + '_angles.' + image_format, format=image_format, dpi=300)
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Image pre-processing helpers
# ─────────────────────────────────────────────────────────────────────────────

def cleanup_labels(img):
    """Remove connected components with ≤ 3 pixels from each label.

    Segmentation algorithms occasionally produce small satellite fragments that
    share a label with a distant cell.  This function discards them while
    preserving all larger regions of each label.
    """
    cleaned = np.zeros_like(img, dtype=np.int64)
    for label in range(1, np.max(img) + 1):
        mask          = (img == label)
        labeled_cc, n = ndimage.label(mask)
        sizes         = ndimage.sum(mask, labeled_cc, range(n + 1))
        keep          = sizes > 3
        keep[0]       = False  # background component
        cleaned      += np.int64(keep[labeled_cc]) * label
    return cleaned


def dilate_labels(img):
    """Grow each label by one pixel using a 3×3 structuring element.

    Conflicts are resolved by giving precedence to whichever label was already
    present in the dilated image at that position.

    Note: this function exists as a utility but is not called in the default
    analysis pipeline.
    """
    dilated   = np.zeros_like(img)
    structure = np.ones((3, 3), dtype=bool)
    for label in range(1, np.max(img) + 1):
        grown    = ndimage.binary_dilation(img == label, structure=structure)
        conflict = dilated > 0
        dilated += (grown & ~conflict) * label
    return dilated


# ─────────────────────────────────────────────────────────────────────────────
# Main analysis pipeline
# ─────────────────────────────────────────────────────────────────────────────

def parameter_analysis(
    in_silico,
    use_fluid_binary=False,
    use_micron_bins=False,
    image_micron_bins=30,
    image_bins=1,
    ccc_threshold=0,
    only_tcj=False,
    folderpath='./segmentations/',
    image_format='svg',
    threshold_distance=3,
    in_silico_crop=[100, 100, 100, 100],
):
    """Run the full FACCC parameter extraction pipeline.

    Iterates over all Cellpose segmentation masks (``*cp_masks*``) found in
    *folderpath*, optionally splits each image into spatial bins, and computes
    a comprehensive set of tissue parameters.  Results are saved to
    ``./results.xlsx`` and to per-cell / per-pocket CSVs.

    Parameters
    ----------
    in_silico : bool
        True for in-silico tissues (tighter CCC threshold, 3×3 label dilation,
        and border cropping are applied).  False for experimental images.
    use_fluid_binary : bool
        When True, a co-registered binary fluid mask (``*_IF_binary.tif``)
        is used to zero out fluid pixels before analysis.
    use_micron_bins : bool
        Split each image into spatial bins based on physical size.  Requires
        the image filename to encode the FOV height in microns, e.g.
        ``..._100x100_...png``.
    image_micron_bins : int
        Bin height in microns when *use_micron_bins* is True (default 30 µm).
    image_bins : int
        Number of equal-height strips to analyse independently when
        *use_micron_bins* is False (default 1 = whole image at once).
    ccc_threshold : float
        Cell-contact coverage below which a cell is considered in contact with
        fluid.  Set automatically to 0.95 (in-silico) or 0.68 (experimental)
        when left at 0.
    only_tcj : bool
        If True, count only tri-cellular junctions (TCJ; exactly 3 labels).
        If False, count all multi-cellular junctions (MCJ; ≥ 3 labels) and
        report TCJs separately for the triangle-closure fraction.
    folderpath : str
        Directory containing the segmentation PNG files.
    image_format : str
        Output figure format ('svg' or 'png').
    threshold_distance : int
        Pixel radius used to merge nearby junction detections.
    in_silico_crop : [top, bottom, left, right]
        Border crop in pixels applied to in-silico images to remove boundary
        artefacts introduced by the simulation domain edges.
    """
    if ccc_threshold == 0:
        ccc_threshold = 0.95 if in_silico else 0.68

    plt.ioff()
    dfsavepath = './results.xlsx'

    # Create all required output sub-directories.
    for subdir in ['results', 'results/Adjmatrix', 'results/per_cell_csvs', 'results/per_pocket_csvs']:
        os.makedirs(os.path.join(folderpath, subdir), exist_ok=True)

    cp_masks = sorted([f for f in os.listdir(folderpath) if 'cp_masks' in f])

    # Per-image accumulator lists (one entry per bin per image).
    names                   = []
    ccc_mean                = [];  ccc_sem                 = []
    area_mean               = [];  area_sem                = []
    perimeter_mean          = [];  perimeter_sem           = []
    circularity_mean        = [];  circularity_sem         = []
    orientation_mean        = [];  orientation_sem         = []
    major_axis_mean         = [];  major_axis_sem          = []
    minor_axis_mean         = [];  minor_axis_sem          = []
    total_fluid             = []
    fluid_mean              = [];  fluid_sem               = []
    fluidfraction           = []
    thresholdedratios       = []
    N_of_triangles          = []
    per_trg_closed          = []
    pockets_no              = []
    mcj_per_cell_mean       = [];  mcj_per_cell_sem        = []
    mcj_per_cell_total_mean = [];  mcj_per_cell_total_sem  = []
    angles_mean             = [];  angles_sem              = []
    N_cells                 = [];  avK                     = []
    GRC                     = [];  GRC2                    = []
    cells_in_contact_long   = []
    mcjs_no_lst             = []

    for cp_mask in tqdm(cp_masks):
        print(cp_mask)
        labeled_image_for_bin = skio.imread(os.path.join(folderpath, cp_mask), as_gray=True)

        # Rescale float images output by some segmenters from [0, 1] to uint8.
        if labeled_image_for_bin.dtype == np.float64:
            labeled_image_for_bin = (labeled_image_for_bin * 255).astype(np.uint8)
            labeled_image_for_bin -= np.min(labeled_image_for_bin)

        labeled_image_for_bin = cleanup_labels(labeled_image_for_bin)

        # Zero out pixels belonging to the fluid channel if a binary mask is provided.
        if use_fluid_binary:
            fluid_binary = cp_mask[:-12] + 'IF_binary.tif'
            fluid_mask   = np.where(skio.imread(os.path.join(folderpath, fluid_binary), as_gray=True) != 0)
            labeled_image_for_bin[fluid_mask] = 0

        # For in-silico images: relabel contiguous regions and crop the border to
        # remove artefacts introduced by the simulation domain boundary.
        if in_silico:
            relabeled = np.zeros_like(labeled_image_for_bin)
            for new_val, old_val in enumerate(np.unique(labeled_image_for_bin)):
                relabeled[labeled_image_for_bin == old_val] = new_val
            relabeled = measure.label(relabeled, connectivity=1).astype(int)
            t, b, l, r = in_silico_crop
            labeled_image_for_bin = relabeled[t:-(b + 1), l:-(r + 1)]

        # Determine spatial binning parameters.
        if use_micron_bins:
            y_microns  = int(cp_mask.split('x')[1].split('_')[0])
            bin_px     = labeled_image_for_bin.shape[0] / y_microns * image_micron_bins
            image_bins = math.floor(y_microns / image_micron_bins)
        else:
            bin_px = labeled_image_for_bin.shape[0] / image_bins

        for image_bin in range(image_bins):
            y_min = int(bin_px * (image_bins - image_bin - 1))
            y_max = min(int(bin_px * (image_bins - image_bin)), labeled_image_for_bin.shape[0])
            labeled_image = labeled_image_for_bin[y_min:y_max + 1]

            # ── Cell morphology (border cells excluded) ───────────────────────
            # Cells touching the image edge are excluded from shape statistics
            # because their segmented area and perimeter are artificially truncated.
            border_labels  = np.unique(np.concatenate([
                labeled_image[0, :], labeled_image[-1, :],
                labeled_image[:, 0], labeled_image[:, -1],
            ]))
            woborderlabels = labeled_image.copy()
            for lbl in border_labels:
                woborderlabels[labeled_image == lbl] = 0

            interior_labels, interior_counts = np.unique(
                woborderlabels[woborderlabels != 0], return_counts=True)

            area_mean.append(np.mean(interior_counts))
            area_sem.append(sem(interior_counts))

            circ_lst      = []
            perimeter_lst = []
            for lbl in interior_labels:
                region = woborderlabels == lbl
                perim  = measure.perimeter(region)
                perimeter_lst.append(perim)
                circ_lst.append(4 * np.pi * np.sum(region) / perim**2)

            perimeter_mean.append(np.mean(perimeter_lst));    perimeter_sem.append(sem(perimeter_lst))
            circularity_mean.append(np.mean(circ_lst));       circularity_sem.append(sem(circ_lst))

            orientation_lst = []
            major_axis_lst  = []
            minor_axis_lst  = []
            for prop in measure.regionprops(woborderlabels):
                if prop.label != 0:
                    orientation_lst.append(prop.orientation)
                    major_axis_lst.append(prop.axis_major_length)
                    minor_axis_lst.append(prop.axis_minor_length)

            orientation_mean.append(np.mean(orientation_lst));  orientation_sem.append(sem(orientation_lst))
            major_axis_mean.append(np.mean(major_axis_lst));    major_axis_sem.append(sem(major_axis_lst))
            minor_axis_mean.append(np.mean(minor_axis_lst));    minor_axis_sem.append(sem(minor_axis_lst))

            # ── Cell-Cell Contact (CCC) and adjacency matrix ──────────────
            labels = np.unique(labeled_image)
            labels = labels[labels != 0]

            fluid_image    = (labeled_image == 0).astype(int)
            labeled_fluids = measure.label(fluid_image, background=0)

            ccc_lst    = []
            adj_matrix = []
            label_lst  = []
            border_arr = []

            for lbl in labels:
                label_lst.append(lbl)
                mask = np.uint8(labeled_image == lbl)

                # Build a 1-pixel-wide ring around the cell boundary using two dilations.
                ring_inner = dilation(mask, footprint=np.ones((3, 3), dtype=np.uint8))
                ring_outer = dilation(mask, footprint=np.ones((5, 5), dtype=np.uint8))
                outline    = ring_outer - ring_inner

                # Add the outline to the labeled image; pixels that did not change
                # are outside the ring and zeroed out.
                overlay = labeled_image + outline
                overlay[labeled_image == overlay] = 0

                touching, pixel_counts = np.unique(overlay, return_counts=True)
                if 0 in touching:
                    pixel_counts = pixel_counts[touching != 0]
                    touching     = touching[touching != 0]

                # The first remaining entry represents fluid-adjacent outline pixels
                # (where the original labeled_image value was 0, shifted to 1 by +outline).
                total_border = pixel_counts[0]

                # Remove the fluid-contact entry; remaining entries are cell–cell contacts.
                pixel_counts = pixel_counts[touching != 1]
                touching     = touching[touching != 1]

                some_border = 0
                for k, neighbour in enumerate(touching):
                    adj_matrix   = np.append(adj_matrix, [lbl, neighbour - 1, pixel_counts[k]])
                    total_border += pixel_counts[k]
                    some_border  += pixel_counts[k]

                border_arr.append(some_border / total_border)
                # CCC: fraction of the cell boundary that is in contact with other cells.
                ccc_lst.append(sum(pixel_counts) / len(np.where(outline == 1)[0]))

            bp_df = pd.DataFrame({'Label': label_lst, 'CCC': ccc_lst})
            name  = cp_mask[:-13] + '_bin' + str(image_bin)
            names.append(name)

            # ── Label dilation to close thin inter-cell gaps ──────────────────
            # Experimental images: minimal 1×1 dilation (no gap closure).
            # In-silico images: 3×3 dilation to close the simulated cell-wall gaps.
            new_img   = np.zeros(labeled_image.shape)
            footprint = np.ones((3, 3) if in_silico else (1, 1), dtype=np.uint8)
            for lbl in labels:
                mask = np.uint8(labeled_image == lbl)
                grown = dilation(mask, footprint=footprint)
                new_img[grown == 1] = lbl
                new_img[mask  == 1] = lbl  # original pixels take precedence

            labeled_image = new_img.astype(int)

            # Remove any labels that became empty after contiguous relabelling.
            lost       = np.setdiff1d(
                np.union1d(np.unique(labeled_image)[np.unique(labeled_image) != 0], labels),
                np.intersect1d(np.unique(labeled_image)[np.unique(labeled_image) != 0], labels),
            )
            bp_df      = bp_df[~bp_df['Label'].isin(lost)]
            border_arr = np.delete(border_arr, np.where(np.isin(labels, lost)))
            labels     = np.delete(labels,     np.where(np.isin(labels, lost)))

            # ── Fluid pocket statistics ───────────────────────────────────────
            fluid_image    = (labeled_image == 0).astype(int)
            total_fluid.append(int(np.sum(fluid_image)))
            labeled_fluids = measure.label(fluid_image, background=0)

            # Exclude the background and the border-touching fluid region (indices 0 and 1).
            pocket_sizes = np.unique(labeled_fluids, return_counts=True)[1][2:]
            fluid_mean.append(np.mean(pocket_sizes))
            fluid_sem.append(sem(pocket_sizes) if len(pocket_sizes) > 1 else 0)
            pockets_no.append(len(pocket_sizes))

            # ── Contact network graph and binary adjacency matrix ─────────────
            edges_df = pd.DataFrame(
                adj_matrix.astype(int).reshape((-1, 3)),
                columns=['CellLabel', 'NeighboringCellLabel', 'BorderPixelCount'],
            )

            # Build a networkx graph using cell centroids as node positions.
            df_centroids = pd.DataFrame(
                [[r.label, *r.centroid[::-1]] for r in measure.regionprops(labeled_image)],
                columns=['Label', 'Centroid.X', 'Centroid.Y'],
            )
            df_centroids['CCC'] = border_arr

            G_nx = nx.Graph()
            for _, row in df_centroids.iterrows():
                G_nx.add_node(int(row['Label']), pos=[row['Centroid.X'], row['Centroid.Y']])

            max_label         = max(max(u, v) for u, v in edges_df[['CellLabel', 'NeighboringCellLabel']].values)
            Binary_adj_matrix = np.zeros((max_label, max_label), dtype=int)

            for u, v in edges_df[['CellLabel', 'NeighboringCellLabel']].values:
                G_nx.add_edge(u, v)
                Binary_adj_matrix[u - 1, v - 1] = 1
                Binary_adj_matrix[v - 1, u - 1] = 1

            np.save(
                os.path.join(folderpath, 'results', 'Adjmatrix', name + '_Adj_matrix.npy'),
                Binary_adj_matrix,
            )

            # Number of triangular cliques = tr(A³) / 6.
            N_of_triangles.append(np.trace(np.linalg.matrix_power(Binary_adj_matrix, 3)) / 6)

            # ── GRC network visualisation ─────────────────────────────────────
            pos      = nx.get_node_attributes(G_nx, 'pos')
            width_arr = np.array([
                edges_df.loc[
                    (edges_df['CellLabel'] == u) & (edges_df['NeighboringCellLabel'] == v),
                    'BorderPixelCount',
                ].values[0] / 20
                for u, v in G_nx.edges
            ])

            # Colour cell regions by their CCC value for the background image.
            ccc_image = labeled_image.copy().astype(float)
            for _, row in df_centroids.iterrows():
                ccc_image[labeled_image == int(row['Label'])] = row['CCC'] * 100

            Giant_Cluster, Second_Giant_Cluster, PC, bernat_values = getClusters(
                edges_df=edges_df, N_cells_total=len(labels))

            # Assign edge colours: GRC = red, 2GRC = blue, percolation = yellow, rest = black.
            for u, v in G_nx.edges():
                if (u, v) in PC or (v, u) in PC:
                    color = 'y'
                elif (u, v) in Giant_Cluster or (v, u) in Giant_Cluster:
                    color = 'red'
                elif (u, v) in Second_Giant_Cluster or (v, u) in Second_Giant_Cluster:
                    color = 'blue'
                else:
                    color = 'black'
                G_nx[u][v]['color'] = color

            G_nx.remove_nodes_from(set(G_nx.nodes()) - set(pos.keys()))
            plt.figure(figsize=(6, 6))
            plt.imshow(ccc_image, cmap=mpl.colormaps['Greys'])
            nx.draw(G_nx, pos, with_labels=True, node_size=50, node_color='lightblue',
                    font_size=5, font_weight='bold', width=width_arr,
                    edge_color=[G_nx[u][v]['color'] for u, v in G_nx.edges()])
            plt.savefig(
                os.path.join(folderpath, 'results', name + '_GRC.' + image_format),
                format=image_format, dpi=300,
            )
            plt.close()

            # ── CCC statistics ────────────────────────────────────────────────
            ccc_arr = bp_df['CCC'].values
            ccc_mean.append(np.mean(ccc_arr))
            ccc_sem.append(np.std(ccc_arr) / np.sqrt(len(ccc_arr)))

            # ── Fluid fraction ────────────────────────────────────────────────
            fluidfraction.append(np.sum(fluid_image) / labeled_image.size)

            # ── Multi/tri-cellular junction detection ─────────────────────────
            # A junction pixel is any position where at least 3 distinct cell labels
            # appear in its 3×3 neighbourhood.
            junction_map = generic_filter(labeled_image, count_unique_labels, size=3)
            if only_tcj:
                junctions_coordinates = np.where(junction_map == 3)
            else:
                junctions_coordinates     = np.where(junction_map > 2)
                junctions_coordinates_TCJ = np.where(junction_map == 3)
                coordss_TCJ               = np.column_stack(junctions_coordinates_TCJ)
                averaged_coords_TCJ       = average_close_points(coordss_TCJ, threshold_distance)

            coordss         = np.column_stack(junctions_coordinates)
            averaged_coords = average_close_points(coordss, threshold_distance)

            # Count how many junctions each cell participates in (11×11 px search window).
            junctions_labels = []
            for cy, cx in averaged_coords:
                patch = labeled_image[cy - 5:cy + 6, cx - 5:cx + 6]
                junctions_labels.extend(np.unique(patch))
            junctions_labels = [v for v in junctions_labels if v != 0]

            unique_mcjs = np.unique(junctions_labels, return_counts=True)[1]
            mcj_per_cell_mean.append(np.mean(unique_mcjs))
            mcj_per_cell_sem.append(sem(unique_mcjs))

            n_total_padding = len(np.unique(labeled_image) - 1)
            n_total_cells   = len(np.unique(labeled_image)) - 1
            padded          = np.pad(unique_mcjs, (0, n_total_padding - len(unique_mcjs)), constant_values=0)
            mcj_per_cell_total_mean.append(np.mean(padded))
            mcj_per_cell_total_sem.append(sem(padded))

            # Fraction of all cells that participate in at least one junction.
            proximity_patches = [
                labeled_image[
                    max(0, x - 1):min(labeled_image.shape[0], x + 2),
                    max(0, y - 1):min(labeled_image.shape[1], y + 2),
                ]
                for x, y in zip(*junctions_coordinates)
            ]
            if proximity_patches:
                flat   = np.concatenate([p.flatten() for p in proximity_patches])
                unique = np.unique(flat)
                cells_in_contact_long.append(len(unique[unique != 0]) / n_total_cells)
            else:
                cells_in_contact_long.append(0)

            # ── CCC-thresholded contact fraction ──────────────────────────────
            thresholdedratios.append((bp_df['CCC'] < ccc_threshold).sum() / len(bp_df))

            # ── Contact angle alpha at fluid pockets ──────────────────────────
            # Build a composite image that uniquely encodes (cell, fluid pocket) pairs:
            # pixels belonging to fluid have value = fluid_label + max_cell_label,
            # so all fluid values exceed max_cell_label while remaining distinguishable.
            max_cell_label = int(np.max(labeled_image))
            fluid_offset   = np.where(labeled_fluids != 0, max_cell_label, 0)
            combined       = labeled_fluids + fluid_offset + labeled_image

            all_cell_labels = np.unique(labeled_image)
            coords   = []
            touchers = []

            for lbl in all_cell_labels:
                other = all_cell_labels[all_cell_labels != lbl]
                for i, j in np.argwhere(labeled_image == lbl):
                    region  = combined[i - 1:i + 2, j - 1:j + 2]
                    ulabels = np.unique(region)
                    # Keep only pixels where this cell, another cell, and a fluid pocket all meet.
                    if (
                        lbl in ulabels
                        and len(np.intersect1d(ulabels, other)) >= 1
                        and np.max(ulabels) > max_cell_label
                        and np.sum(region > max_cell_label) > 1
                    ):
                        coords.append((i, j))
                        touchers.append(ulabels)

            # Group pixels that share the same touching-label signature and take their median.
            duplicate_touchers = find_duplicate_arrays_with_indexes(touchers)
            median_coordinates = [
                find_median_coordinate([coords[k] for k in idxs])
                for idxs in duplicate_touchers.values()
            ]

            inters3  = []
            str_arr  = []
            centers  = []
            x_coords, y_coords = zip(*median_coordinates)

            # For each candidate junction centre, shrink the search radius until two
            # arm endpoints (inters_low and inters_high) can be located.
            offset        = 1
            radius        = 10
            region_offset = 1

            for posi in range(len(x_coords)):
                centeri     = (x_coords[posi], y_coords[posi])
                inters_high = []
                inters_low  = []
                new_radius  = radius

                while len(inters_high) < 1 or len(inters_low) < 1:
                    new_radius -= 1
                    coori = get_coordinates_in_radius(centeri, new_radius)

                    orig = np.unique(combined[
                        centeri[0] - offset:centeri[0] + offset + 1,
                        centeri[1] - offset:centeri[1] + offset + 1,
                    ])
                    orig = orig[orig != 0]

                    if len(orig) == 3:
                        for coor in coori:
                            reg = combined[
                                coor[0] - region_offset:coor[0] + region_offset + 1,
                                coor[1] - region_offset:coor[1] + region_offset + 1,
                            ]
                            reg_labels = np.unique(reg)
                            for orig_lbl in orig:
                                if (
                                    orig_lbl in reg_labels
                                    and np.max(reg_labels) > max_cell_label
                                    and len(reg_labels) > 1
                                    and np.max(orig) in reg
                                ):
                                    new_coor = find_median_coordinate(
                                        find_pixels_with_two_different_neighbors(reg, 1))
                                    new_coor = (
                                        coor[0] + new_coor[0] - region_offset,
                                        coor[1] + new_coor[1] - region_offset,
                                    )
                                    if orig[-2] == orig_lbl:
                                        inters_high.append(new_coor)
                                    if np.min(orig) == orig_lbl:
                                        inters_low.append(new_coor)

                    if new_radius < 3:
                        break

                if inters_low and inters_high:
                    ih = find_median_coordinate(inters_high)
                    il = find_median_coordinate(inters_low)
                    if ih != il:
                        try:
                            angle_deg, angle_cos = calculate_angle(
                                il, (x_coords[posi], y_coords[posi]), ih)
                            str_arr.append('0.000' if math.isnan(angle_deg) else str(round(angle_cos, 3)))
                            inters3.extend([il, ih])
                            centers.append((x_coords[posi], y_coords[posi]))
                        except ValueError:
                            pass

            angles_mean.append(np.mean(np.array(str_arr, dtype=float)))
            angles_sem.append(np.std(np.array(str_arr, dtype=float)) / np.sqrt(len(str_arr)))
            N_cells.append(bernat_values[0])
            avK.append(bernat_values[1])
            GRC.append(bernat_values[2])
            GRC2.append(bernat_values[3])

            junctions_list = list(zip(*junctions_coordinates))
            mcjs_no_lst.append(len(averaged_coords))

            # Fraction of triangular cliques in the contact graph that are
            # closed by a real TCJ / MCJ in the image.
            n_triangles = np.trace(np.linalg.matrix_power(Binary_adj_matrix, 3)) / 6
            if n_triangles == 0:
                per_trg_closed.append(0)
            elif only_tcj:
                per_trg_closed.append(len(averaged_coords) / n_triangles)
            else:
                per_trg_closed.append(len(averaged_coords_TCJ) / n_triangles)

            get_better_coordinates(
                junctions_list, inters3, centers, str_arr,
                labeled_image=labeled_image, key=name,
                plot=True, folderpath=folderpath,
                averaged_coords=averaged_coords,
                ori=False, radiuspx=15, image_format=image_format,
            )

            # ── Per-cell CSV ──────────────────────────────────────────────────
            per_cell_records = []
            for prop in measure.regionprops(labeled_image):
                perim = prop.perimeter
                circ  = 4 * np.pi * prop.area / perim**2 if perim > 0 else 0
                per_cell_records.append({
                    'Label':       prop.label,
                    'Area':        prop.area,
                    'Perimeter':   perim,
                    'Circularity': circ,
                    'Border':      'no' if prop.label in interior_labels else 'yes',
                })
            (pd.DataFrame(per_cell_records)
             .query('Label != 0')
             .to_csv(os.path.join(folderpath, 'results', 'per_cell_csvs', name + '.csv'), index=False))

            # ── Per-pocket CSV ────────────────────────────────────────────────
            pocket_border_labels = np.unique(np.concatenate([
                labeled_fluids[0, :], labeled_fluids[-1, :],
                labeled_fluids[:, 0], labeled_fluids[:, -1],
            ]))
            interior_pockets = set(np.unique(labeled_fluids)) - set(pocket_border_labels) - {0}
            per_pocket_records = []
            for prop in measure.regionprops(labeled_fluids):
                perim = prop.perimeter
                circ  = 4 * np.pi * prop.area / perim**2 if perim > 0 else 0
                per_pocket_records.append({
                    'Label':       prop.label,
                    'Area':        prop.area,
                    'Perimeter':   perim,
                    'Circularity': circ,
                    'Border':      'no' if prop.label in interior_pockets else 'yes',
                })
            (pd.DataFrame(per_pocket_records)
             .query('Label != 0')
             .to_csv(os.path.join(folderpath, 'results', 'per_pocket_csvs', name + '.csv'), index=False))

            # ── Aggregate results table ────────────────────────────────────────
            # Written after every bin so that partial results are always on disk.
            jt = 'tcj' if only_tcj else 'mcj'
            df = pd.DataFrame({
                'embryo':                           names,
                'ccc_mean':                         ccc_mean,
                'ccc_sem':                          ccc_sem,
                'fluidfraction_FACCC':              fluidfraction,
                'num_triangles':                    N_of_triangles,
                'per_triangles_closed':             per_trg_closed,
                f'cellswith{jt}':                   cells_in_contact_long,
                f'{jt}s_no':                        mcjs_no_lst,
                f'{jt}_per_cell_mean':              mcj_per_cell_mean,
                f'{jt}_per_cell_sem':               mcj_per_cell_sem,
                f'{jt}_per_cell_total_mean':        mcj_per_cell_total_mean,
                f'{jt}_per_cell_total_sem':         mcj_per_cell_total_sem,
                'cellsincontactwithfluidthresholded': thresholdedratios,
                'alpha_mean_FACCC':                 angles_mean,
                'alpha_sem_FACCC':                  angles_sem,
                'N_cells_FACCC':                    N_cells,
                'avK_FACCC':                        avK,
                'GRC_FACCC':                        GRC,
                'GRC2_FACCC':                       GRC2,
                'cell_area_mean':                   area_mean,
                'cell_area_sem':                    area_sem,
                'cell_perimeter_mean':              perimeter_mean,
                'cell_perimeter_sem':               perimeter_sem,
                'circularity_mean':                 circularity_mean,
                'circularity_sem':                  circularity_sem,
                'pocket_size_mean':                 fluid_mean,
                'pocket_size_sem':                  fluid_sem,
                'pockets_no':                       pockets_no,
                'total_fluid_FACCC':                total_fluid,
                'cell_orientation_mean':            orientation_mean,
                'cell_orientation_sem':             orientation_sem,
                'cell_major_axis_mean':             major_axis_mean,
                'cell_major_axis_sem':              major_axis_sem,
                'cell_minor_axis_mean':             minor_axis_mean,
                'cell_minor_axis_sem':              minor_axis_sem,
            })
            df.to_excel(dfsavepath)


# ─────────────────────────────────────────────────────────────────────────────
# In-silico utility functions
# ─────────────────────────────────────────────────────────────────────────────

def create_in_silico_df():
    """Append simulation parameters parsed from sample names in results.xlsx.

    Extracts the target contact angle (alpha), system size (N), and fluid
    fraction from the filename convention and adds them as columns.
    Output is saved to ``results_with_parameters.xlsx``.
    """
    df = pd.read_excel('./results.xlsx')
    df['alpha_name'] = df['embryo'].apply(
        lambda n: n.split('alpha')[1][0] + '.' + n.split('alpha')[1][2:5])
    df['n_name']     = df['embryo'].apply(lambda n: n.split('_')[0][1:])
    df['frac_name']  = df['embryo'].apply(
        lambda n: n.split('frac')[1][0] + '.' + n.split('frac')[1][2:5])
    df['relative_fluid'] = df['pocket_size_mean'] / df['total_fluid_FACCC']
    df.iloc[:, 1:].to_excel('./results_with_parameters.xlsx')


def rename_files(folder_path):
    """Rename in-silico PostScript output files to the expected naming convention.

    Renames files from ``Fcells_..._Alpha<val>_N<val>_iteration<val>.ps``
    to ``<N>_alpha<val>_it<val>.ps`` so that `in_silico_segmentation` can
    parse sample parameters from the filename.

    Parameters
    ----------
    folder_path : str
        Directory containing raw ``Fcells_*`` PostScript files.
    """
    for filename in os.listdir(folder_path):
        if not filename.startswith('Fcells_'):
            continue
        parts       = filename.split('_')
        N_value     = parts[3]
        alpha_value = parts[2][5:] if parts[2][5:] != 'aa' else '1'
        it_value    = parts[4][9:]
        new_name    = f'{N_value}_alpha{alpha_value}_it{it_value}.ps'
        os.rename(os.path.join(folder_path, filename), os.path.join(folder_path, new_name))


def convert_ps_to_png(ps_file_path, output_folder, dpi):
    """Convert a PostScript file to a greyscale PNG using Ghostscript.

    Parameters
    ----------
    ps_file_path  : str   Path to the input ``.ps`` file.
    output_folder : str   Output directory.
    dpi           : int   Rendering resolution.

    Returns
    -------
    str   Path of the written PNG file.
    """
    base     = os.path.splitext(os.path.basename(ps_file_path))[0]
    png_path = os.path.join(output_folder, base + '_cp_masks.png')
    subprocess.call(['gs', '-sDEVICE=pnggray', f'-r{dpi}', '-o', png_path, ps_file_path])
    return png_path


def in_silico_segmentation(
    ps_folder='./ps/',
    convexity_threshold=0.925,
    circularity_threshold=0.600,
    image_dpi=300,
    area_high_quantile=1.05,
):
    """Convert in-silico PostScript geometry files to labelled segmentation masks.

    Each ``.ps`` file is:
      1. Rasterised to a greyscale PNG via Ghostscript.
      2. Skeletonised and tightly cropped.
      3. Connected-component labelled.
      4. Filtered: components outside the expected cell-size range or below the
         convexity / circularity thresholds are removed.
      5. Saved in-place as a PNG ready for `parameter_analysis`.

    Parameters
    ----------
    ps_folder             : str    Directory containing ``.ps`` input files.
    convexity_threshold   : float  Minimum solidity (area / convex_area).
    circularity_threshold : float  Minimum circularity (4π·A / P²).
    image_dpi             : int    Ghostscript rendering resolution.
    area_high_quantile    : float  Multiplier on the 80th-percentile area for
                                   the upper acceptable cell-size bound.
    """
    output_folder = './segmentations/'
    os.makedirs(output_folder, exist_ok=True)

    for ps_file in os.listdir(ps_folder):
        png_path = convert_ps_to_png(os.path.join(ps_folder, ps_file), output_folder, dpi=image_dpi)

        circles = skio.imread(png_path, as_gray=True).astype(int)
        if circles.ndim > 2:
            circles = circles[:, :, 0]

        skel = skeletonize(np.where(circles == 255, 0, 1))

        # Crop tightly around the cell-skeleton bounding box.
        ys, xs = np.where(skel == 1)
        try:
            skel_crop = skel[ys.min() - 4:ys.max() + 5, xs.min() - 4:xs.max() + 5]
            skio.imsave(png_path, skel_crop)
        except (IndexError, ValueError):
            continue

        skel_rev = np.where(skel_crop == 1, 0, 1)
        labeled  = measure.label(skel_rev, connectivity=1)
        props    = measure.regionprops(labeled)

        areas     = [p.area for p in props]
        area_high = np.quantile(areas, 0.8) * area_high_quantile
        area_low  = 0.4 * area_high

        for prop in props:
            too_big    = prop.area > area_high
            too_small  = prop.area < area_low
            not_convex = (prop.area / prop.convex_area) < convexity_threshold
            not_round  = (4 * np.pi * prop.area / prop.perimeter**2) < circularity_threshold
            if too_big or too_small or not_convex or not_round:
                labeled[labeled == labeled[prop.coords[0, 0], prop.coords[0, 1]]] = 0

        plt.imsave(png_path, labeled)
