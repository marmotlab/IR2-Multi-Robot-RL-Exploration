#######################################################################
# Name: parameter.py
# Inference parameters.
#######################################################################

import sys
sys.modules['TRAINING'] = False           # False = Inference Testing                                                                                            

# --- MAPS --- #
# TODO: Set train set
TEST_SET_NAME = "hybrid"        # "hybrid", "corridor", "complex", 
TEST_SET_DIR = "DungeonMaps/test/" + TEST_SET_NAME

# Easier maps
if TEST_SET_NAME == "hybrid" or TEST_SET_NAME == "corridor":
    MAX_EPS_STEPS=196   
    K_SIZE = 30
    NUM_ROBOTS_MIN=4
    NUM_ROBOTS_MAX=4
    NODE_COORDS_SCALING_FACTOR=1/640    
    NODE_UTILITY_SCALING_FACTOR=1/50   
    GLOBAL_GRAPH_NODE_COORDS_THRESH=200       # Num node coords before start to perform graph merger
    NODE_PADDING_SIZE = 99999 

# Complex maps
else:
    MAX_EPS_STEPS=384   
    K_SIZE = 30
    NUM_ROBOTS_MIN=5
    NUM_ROBOTS_MAX=5
    NODE_COORDS_SCALING_FACTOR=1/1000   
    NODE_UTILITY_SCALING_FACTOR=1/50    
    GLOBAL_GRAPH_NODE_COORDS_THRESH=340       # Num node coords before start to perform graph merger  
    NODE_PADDING_SIZE = 99999 

    if TEST_SET_NAME != "complex":
        print("[test_parameter.py] INVALID TEST_SET_NAME!!!")


# --- Test Specific --- #
NUM_TEST = 100                      # Number of runs in this test
NUM_RUN = 1                         # How many times to run this set of tests
SAVE_TRAJECTORY = False             # Do you want to save per-step metrics 
SAVE_LENGTH = False                 # Do you want to save per-episode metrics 
SAVE_GIFS = True                    # Do you want to save GIFs
VIZ_GRAPH_EDGES=True                # Visualize graph edhes (NOTE: Will be very slow)
VIZ_GRAPH_EDGES_GROUND_TRUTH=False  # Visualize graph edhes (NOTE: Will be very slow)

# --- GENERAL --- #
USE_GPU = True
NUM_GPU = 1
NUM_META_AGENT = 1  # Number of parallel sims
FOLDER_NAME = 'mar_inference' 
MODEL_DIR = f'model/stage2'
GIFS_DIR = f'{FOLDER_NAME}/test_results/gifs'
MODEL_PATH = MODEL_DIR + '/checkpoint.pth' 
trajectory_path = f'{FOLDER_NAME}/test_results/trajectory'
length_path = f'{FOLDER_NAME}/test_results/length'
log_path = f'{FOLDER_NAME}/test_results/log'

# --- RL Params --- # 
EMBEDDING_DIM = 128
INPUT_DIM = 6

# --- Sensor Model --- # 
# SS Reference: https://hal.science/hal-03365129/document
SENSOR_RANGE=80
UTILITY_CALC_RANGE=70  
USE_SIGNAL_STRENGTH_NOT_PROXIMITY=True
PROXIMITY_COMMS_RANGE_MIN=30
PROXIMITY_COMMS_RANGE_MAX=150
SS_P_T=-20
SS_THRESH=-70
SS_GAMMA=2
SS_GAMMA_OBST=4
SS_DIST_O=35
SS_PL_O=31
SS_XG_MIN=0
SS_XG_MAX=13
SS_K_MIN=0
SS_K_MAX=13

# --- Graph Params (General) --- # 
NUM_DENSE_COORDS_WIDTH=50                 # How many node coords across width?
NUM_DENSE_COORDS_HEIGHT=50                # How many node coords across height?
CUR_AGENT_KNN_RAD=80                      # How far current agent's node coords can be saved in dense local coords.
OTHER_AGENT_KNN_RAD=20                    # How far other agent's node coords can be saved in dense local coords.
GLOBAL_GRAPH_KNN_RAD=160                  # Global graph max edge length (Should be about 2x sensor range)
GLOBAL_GRAPH_UNIQUE_RAD=30                # How far apart should graph points be 
GLOBAL_GRAPH_OFFSHOOT_UNIQUE_RAD=40       # How far apart should offshoot points be
GLOBAL_GRAPH_OFFSHOOT_FRONTIER_NODES=6    # Up to how many nodes to add to offshoots of global graph, per step
GLOBAL_GRAPH_OFFSHOOT_MAX_RAD=80          # How far offshoot nodes can be from global graph 

# --- Graph Params (Merger) --- # 
MERGE_GLOBAL_GRAPH=True                             # Whether to merge global graph nodes (with neighboring nodes)
MERGE_GLOBAL_GRAPH_EVERY=10                         # Perform global graph merger every MERGE_GLOBAL_GRAPH_EVERY episode
RAYTRACE_ZERO_UTIL_GLOBAL_NODES_TO_SPARSIFY=False   # Whether to raytrace frontiers (consider collision) to determine zero utility global nodes to sparsify (Else, euclidean dist from frontiers)
GLOBAL_NODES_TO_FRONTIER_AVOID_SPARSE_RAD=120       # [If RAYTRACE_ZERO_UTIL_GLOBAL_NODES_TO_SPARSIFY=False] Radius of node to frontier, to avoid being in sparse process (Preferred: 1.5x sensor range)

# --- Graph Params (Pruning) --- # 
PRUNE_GLOBAL_GRAPH=False                            # Whether to perform pruning of zero utility nodes (not leading to frontier centers)
PRUNE_GLOBAL_GRAPH_EVERY=30                         # Prune zero-utility global graph nodes every PRUNE_GLOBAL_GRAPH_EVERY episodes
MIN_CENTERS_BEFORE_SPARSIFY=1                       # Number of centers before we conduct graph pruning
MAX_UTILITY_TO_SPARSE=0.1                           # Prune any node coords below this threshold, if not on A* path
SPARSIFICATION_CENTERS_KNN_RAD=30                   # How far should frontier centers be spaced apart
SPARSIFY_RETAIN_AGENT_TO_AGENT_ASTAR=True           # Whether to consider agent2agent's Astar paths
SPARSIFY_RETAIN_FRONTIER_TO_FRONTIER_ASTAR=False    # Whether to consider frontier2frontier's Astar paths

# --- Map Delta (Surplus) --- #
MIN_MAP_DELTA_MAP_RATIO=0.01                        # To be multiplied with map's free_space to determine min map area for map_delta computation
MAP_DELTA_NORM_FACTOR=8                             # Divide by map area in script
MAP_DELTA_MIN_CONST=0.1                             # Constant in map_delta equation
REMOVE_POSE_BELIEF_MISSING_COUNT=3                  # Remove pose if not seen for REMOVE_POSE_BELIEF_MISSING_COUNT counts
RENDEZVOUS_ASTAR_DENSIFY_PATH_RAD=30.0              # At least 1 point along astar path every RENDEZVOUS_ASTAR_DENSIFY_PATH_RAD meters
RENDEZVOUS_ASTAR_MAP_DELTA_INFLATION_RAD=50.0       # Insert map-delta val to points RENDEZVOUS_ASTAR_MAP_DELTA_INFLATION_RAD meters around each astar path point
RENDEZVOUS_OWN_POSE_NO_UTIL_RAD=30.0                # Set rendezv util around robot to 0 to encourage movement

# COLORS (for printing)
RED='\033[1;31m'          
GREEN='\033[1;32m'
YELLOW='\033[1;93m'       
NC_BOLD='\033[1m' # Bold, No Color 
NC='\033[0m' # No Color 

