# Imports
import os
import tarfile
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import tqdm

from waymo_open_dataset.wdl_limited.sim_agents_metrics import metric_features
from waymo_open_dataset.wdl_limited.sim_agents_metrics import metrics

from waymo_open_dataset.protos import scenario_pb2
from waymo_open_dataset.protos import map_pb2
from waymo_open_dataset.protos import sim_agents_submission_pb2

from waymo_open_dataset.utils.sim_agents import submission_specs
from waymo_open_dataset.utils.sim_agents import test_utils as sim_agents_test_utils
from waymo_open_dataset.utils.sim_agents import visualizations
from waymo_open_dataset.utils import trajectory_utils

from re import I
from google.protobuf.pyext._message import RepeatedScalarContainer

# for lane
TYPE_UNDEFINED = -1
TYPE_FREEWAY = 1
TYPE_SURFACE_STREET= 2
TYPE_BIKE_LANE= 3

# for roadline
TYPE_UNKNOWN = -1
TYPE_BROKEN_SINGLE_WHITE = 6
TYPE_SOLID_SINGLE_WHITE = 7
TYPE_SOLID_DOUBLE_WHITE = 8
TYPE_BROKEN_SINGLE_YELLOW = 9
TYPE_BROKEN_DOUBLE_YELLOW = 10
TYPE_SOLID_SINGLE_YELLOW = 11
TYPE_SOLID_DOUBLE_YELLOW = 12
TYPE_PASSING_DOUBLE_YELLOW = 13

# for roadedge
TYPE_ROAD_EDGE_BOUNDARY = 15
TYPE_ROAD_EDGE_MEDIAN = 16

# for stopsign
TYPE_STOP_SIGN = 17

# for crosswalk
TYPE_CROSSWALK = 18

# for speed bump
TYPE_SPEED_BUMP = 19


import math

def compute_heading(prev_x, prev_y, curr_x, curr_y):
    """
    This function calculates the heading angle in radians between two points given their x and y coordinates.
    """
    delta_x = curr_x - prev_x
    delta_y = curr_y - prev_y

    # Calculate the heading in radians
    heading_rad = math.atan2(delta_y, delta_x)

    # Adjust heading to the range [-pi, pi]
    if heading_rad > math.pi:
        heading_rad -= 2 * math.pi
    elif heading_rad < -math.pi:
        heading_rad += 2 * math.pi

    return heading_rad

def get_waymo_metrics_info(vector):
    # Change the vector dimension of [K,B,T,F] with F = 4 to F = 6
    # intital F : x,y,vx,vy
    # new F : x,y,z,heading,vx,vy
    z = 0.90
    K,B,T,F = vector.shape
    heading = torch.zeros((K,B,T,2))
    for sim in range(K):
        for obj in range(B):
        for i in range(T-1):
            heading[sim,obj,i,1] = compute_heading(vector[sim][obj][i][0], vector[sim][obj][i][1], vector[sim][obj][i+1][0], vector[sim][obj][i+1][1])
            heading[sim,obj,i,0] = z
        heading[sim,obj,T-1,1] = heading[sim,obj,T-2,1]
        heading[sim,obj,T-1,0] = z
    final_tensor = torch.cat((vector[:,:,:,:2],heading[:,:,:,:],vector[:,:,:,2:]),dim=-1)
  
  return final_tensor



def convert_map_tensor_to_map_scenario(myscenario_map, ego_idx = 0):

    # myscenario_map = scenario_pb2.Scenario()
    for pol in range(768):
        pol_type = int(map_polylines_in[ego_idx,pol,0,6])
        map_feature = myscenario_map.map_features.add()
        map_feature.id = pol
        

        for pnt in range(20):
            
            if map_polylines_mask_in[ego_idx, pol, pnt]:
              x,y,z,_,_,_,_,_,_ =  map_polylines_in[ego_idx, pol, pnt,:]           
              polyline =  map_pb2.MapPoint()
              polyline.x = x
              polyline.y = y
              polyline.z = z

              if pol_type in [TYPE_SURFACE_STREET, TYPE_FREEWAY, TYPE_BIKE_LANE]:
                  lane = map_feature.lane
                  lane.type = pol_type
                  lane.polyline.append(polyline)

              # for roadline
              elif pol_type in [TYPE_BROKEN_SINGLE_WHITE, TYPE_SOLID_SINGLE_WHITE,
                            TYPE_SOLID_DOUBLE_WHITE, TYPE_BROKEN_SINGLE_YELLOW, 
                            TYPE_BROKEN_DOUBLE_YELLOW, TYPE_SOLID_SINGLE_YELLOW,
                            TYPE_SOLID_DOUBLE_YELLOW, TYPE_PASSING_DOUBLE_YELLOW]:

                  if pol_type == TYPE_BROKEN_SINGLE_WHITE:
                      local_pol_type = 1
                  elif pol_type == TYPE_SOLID_SINGLE_WHITE:
                      local_pol_type = 2
                  elif pol_type ==TYPE_SOLID_DOUBLE_WHITE:
                      local_pol_type = 3 
                  elif pol_type ==TYPE_BROKEN_SINGLE_YELLOW:
                      local_pol_type = 4
                  elif pol_type ==TYPE_BROKEN_DOUBLE_YELLOW:
                      local_pol_type = 5
                  elif pol_type ==TYPE_SOLID_SINGLE_YELLOW:
                      local_pol_type = 6
                  elif pol_type ==TYPE_SOLID_DOUBLE_YELLOW:
                      local_pol_type = 7
                  elif pol_type ==TYPE_PASSING_DOUBLE_YELLOW:
                      local_pol_type = 8

                  road_line = map_feature.road_line
                  road_line.type = local_pol_type
                  road_line.polyline.append(polyline)
                  
              # for roadedge
              elif pol_type in [TYPE_ROAD_EDGE_BOUNDARY, TYPE_ROAD_EDGE_MEDIAN]:
                  if pol_type == TYPE_ROAD_EDGE_BOUNDARY:
                      local_pol_type = 1
                  else:
                      local_pol_type = 2
                  # print('elif edge pnt',pnt, pol)
                  road_edge = map_feature.road_edge
                  road_edge.type = local_pol_type
                  road_edge.polyline.append(polyline)

              # for stopsign
              elif pol_type == TYPE_STOP_SIGN:
                  stop_sign  = map_feature.stop_sign 
                  # stop_sign.type = pol_type
                  stop_sign.position.append(polyline)

              # for crosswalk
              elif pol_type == TYPE_CROSSWALK:
                  crosswalk = map_feature.crosswalk
                  crosswalk.polygon.append(polyline)

              # for speed bump
              elif pol_type == TYPE_SPEED_BUMP:
                  speed_bump  = map_feature.speed_bump
                  speed_bump.polygon.append(polyline)


    return myscenario_map


def convert_tensor_to_scenario(batch_dict, ego_agent_num = 0):
    
    scenario = scenario_pb2.Scenario()
    tensor_dict = batch_dict['input_dict']
    # SCENARIO ID
    # Assuming scenario_id is a list, select the first element as the scenario ID
    scenario.scenario_id = tensor_dict['scenario_id'][0] #STRING, size 1x16 lettres

    # TIMESTAMPS
    # Generate the timestamps based on the assumption of a 0.1-second step size
    timestamps_val = [0.1 * i for i in range(91)]
    scenario.timestamps_seconds.extend(timestamps_val) #class 'google.protobuf.pyext._message.RepeatedScalarContainer'

    # CURRENT TIME INDEX
    # Set the current_time_index between 0 and 90 based on the assumption of a 0.1-second step size
    scenario.current_time_index = 10

    # TRACKS
    # track = scenario_pb2.Track()
    obj_nb = 0 #later to change for every objects
    object_ids_one = []
    

    for i in range(len(tensor_dict_input['track_index_to_predict'][:])):
        track = scenario_pb2.Track()
        object_ids_one = tensor_dict_input['obj_ids'][tensor_dict_input['track_index_to_predict'][i]]
        track.id = tensor_dict['obj_ids'][tensor_dict_input['track_index_to_predict'][i]]
        if (tensor_dict['obj_types'][i]== 'TYPE_UNSET'):
          track.object_type = 0
        elif (tensor_dict['obj_types'][i]== 'TYPE_VEHICLE'):
          track.object_type = 1
        elif (tensor_dict['obj_types'][i]== 'TYPE_PEDESTRIAN'):
          track.object_type = 2
        elif (tensor_dict['obj_types'][i]== 'TYPE_CYCLIST'):
          track.object_type = 3
        elif (tensor_dict['obj_types'][i]== 'TYPE_OTHER'):
          track.object_type = 4
        else:
          print('error on object type! ')
          print(tensor_dict['obj_types'][i])


        for ts in range(91):
            state = scenario_pb2.ObjectState()
            tensor_dict_input['center_gt_trajs_src']
            obj_traj = tensor_dict['center_gt_trajs_src']

            state.valid = int(obj_traj[i][ts][9])
            state.center_x = float(obj_traj[i][ts][0])
            state.center_y = float(obj_traj[i][ts][1])
            state.center_z = float(obj_traj[i][ts][2])
            state.length = float(obj_traj[i][ts][3])
            state.width = float(obj_traj[i][ts][4])
            state.height = float(obj_traj[i][ts][5])
            state.heading = float(obj_traj[i][ts][6])
            state.velocity_x = float(obj_traj[i][ts][7])
            state.velocity_y = float(obj_traj[i][ts][8])
            
            track.states.append(state)

        scenario.tracks.append(track)

    scenario.sdc_track_index = ego_agent_num # int(tensor_dict['track_index_to_predict'][0])

    scenario = convert_map_tensor_to_map_scenario(scenario)

    return scenario


def joint_scene_from_states(
    states: tf.Tensor, object_ids: tf.Tensor
    ) -> sim_agents_submission_pb2.JointScene:
    # States shape: (num_objects, num_steps, 4).
    # Objects IDs shape: (num_objects,).

    states = states.numpy()
    simulated_trajectories = []

    for i_object in range(len(object_ids)):
    simulated_trajectories.append(sim_agents_submission_pb2.SimulatedTrajectory(
        center_x=states[i_object, :, 0], center_y=states[i_object, :, 1],
        center_z=states[i_object, :, 2], heading=states[i_object, :, 3],
        object_id=object_ids[i_object]
    ))

    return sim_agents_submission_pb2.JointScene(
        simulated_trajectories=simulated_trajectories)


def scenario_rollouts_from_states(
    scenario: scenario_pb2.Scenario,
    states: tf.Tensor, object_ids: tf.Tensor
    ) -> sim_agents_submission_pb2.ScenarioRollouts:

    # States shape: (num_rollouts, num_objects, num_steps, 4).
    # Objects IDs shape: (num_objects,).
    joint_scenes = []
    for i_rollout in range(states.shape[0]):
    joint_scenes.append(joint_scene_from_states(states[i_rollout], object_ids))

    return sim_agents_submission_pb2.ScenarioRollouts(
        # Note: remember to include the Scenario ID in the proto message.
        joint_scenes=joint_scenes, scenario_id=scenario.scenario_id)



def compute_sim_agents_metric(batch_dict):

    input_dict = batch_dict['input_dict']
    obj_ids_in = input_dict['obj_ids']

    out_dists_ou = batch_dict['pred_trajs']
    mode_probs = batch_dict['pred_scores']

    out_dists = np.transpose(out_dists_ou[:32,:,:,:4],(0,2,1,3))
    eager_tensor = tf.convert_to_tensor(out_dists)

        
    center_id = [obj_ids_in[i] for i in track_index_to_predict_in]
    center_id = np.array(center_id)

    scenes_feature = []
    # get all ego_agent scene metric
    for ego_agent in range(len(track_index_to_predict_in)):

        myscenario = convert_tensor_to_scenario(tensor_dict_input, ego_agent)

        # function preparation for submission
        joint_scene = joint_scene_from_states(eager_tensor[ego_agent, :, :, :], center_id)
        submission_specs.validate_joint_scene(joint_scene, myscenario)

        scenario_rollouts = scenario_rollouts_from_states(
            myscenario, eager_tensor, center_id)

        submission_specs.validate_scenario_rollouts(scenario_rollouts, myscenario)

        single_scene_features = metric_features.compute_metric_features(
        myscenario, joint_scene)

        scenes_feature.append(single_scene_features)

    return scenes_feature