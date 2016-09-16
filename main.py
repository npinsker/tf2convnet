import numpy as np
import numpy.fft as fft
import tensorflow as tf
import graph_util

import argparse
import json
import math
import random

def find_node_path(input_node_name, output_node_name, graph):
  if not graph.get_node(input_node_name):
    raise ValueError('Input node %s not in node dictionary.' % input_node)
  if not graph.get_node(output_node_name):
    raise ValueError('Output node %s not in node dictionary.' % output_node)

  def search_for_node(cur_node_proto, target_node_name, graph):
    # We search backwards through the operation graph to find the input node.
    if cur_node_proto.name == target_node_name:
      return [target_node_name]
    for prev_node in cur_node_proto.input:
      path = search_for_node(graph.get_node(prev_node), target_node_name, graph)
      if path:
        return path + [cur_node_proto.name]

  return search_for_node(graph.get_node(output_node_name), input_node_name, graph)

def get_shape(node):
  shape = []
  for dim in node.attr['_output_shapes'].list.shape[0].dim:
    if dim.size != -1:
      shape += [dim.size]
  while len(shape) < 3:
    shape += [1]
  return shape

def array_to_json(arr):
  json_arr = {'sx':1, 'sy':1, 'depth':len(arr), 'w':{}}
  for i in range(len(arr)):
    json_arr['w'][str(i)] = float(arr[i])
  return json_arr

def get_tensor_value(node_name, checkpoint_reader):
  if node_name.endswith('/read'):
    node_name = node_name[:-5]
  return checkpoint_reader.get_tensor(node_name)

def build_affine_matrix(matrix_node, add_node, prev_node_name, checkpoint_reader):
  variable_node_name = matrix_node.input[0]
  if variable_node_name == prev_node_name:
    variable_node_name = matrix_node.input[1]
  variable_tensor = get_tensor_value(variable_node_name, checkpoint_reader)
  bias_tensor = None
  if add_node:
    for prev_node in add_node.input:
      if prev_node != matrix_node.name:
        bias_tensor = get_tensor_value(prev_node, checkpoint_reader)
        break
  else:
    print 'warning: No bias tensor found for node %s!' % matrix_node.name
    bias_tensor = np.array([0] * variable_tensor.shape[0])

  print 'Building FC layer with dimensions', variable_tensor.shape, '...'
  fc_array = {'layer_type': 'fc',
               'out_depth': 1,
               'out_sx': 1,
               'out_sy': 1,
               'biases': array_to_json(bias_tensor),
               'filters': {}}
  for col_index in range(variable_tensor.shape[1]):
    fc_array['filters'][str(col_index)] = array_to_json(variable_tensor[:,col_index])

  return fc_array

def generate_model_json(args):
  INPUT_NODE_NAME = 'Placeholder'
  OUTPUT_NODE_NAME = 'Relu'

  layers = []
  marker = 0
  reader = tf.train.NewCheckpointReader(args.checkpoint_file[0])
  sess = tf.Session()

  saver = tf.train.import_meta_graph(args.meta_file[0])
  saver.restore(sess, args.checkpoint_file[0])

  graph_def = tf.get_default_graph().as_graph_def(add_shapes=True)
  graph = graph_util.TFGraph(graph_def)
  reader = tf.train.NewCheckpointReader(args.checkpoint_file[0])

  path = []
  if not args.input_node or not args.output_node:
    print 'Input and/or output node not supplied! Autodetecting path.'
    path = graph_util.find_longest_path(graph)
    print 'Autodetected path %s' % path
  else:
    path = find_node_path(args.input_node[0], args.output_node[0], graph)
    print 'Found path %s' % path

  while marker < len(path):
    node_proto = graph.get_node(path[marker])
    if node_proto.op == 'Placeholder':
      shape = get_shape(node_proto)
      layers += [{'layer_type': 'input',
                  'out_depth': shape[0],
                  'out_sx': shape[1],
                  'out_sy': shape[2]}]
    elif node_proto.op == 'MatMul':
      if marker < len(path) - 1 and graph.get_node(path[marker+1]).op == 'Add':
        layers += [build_affine_matrix(node_proto, graph.get_node(path[marker+1]),
                                       path[marker-1], reader)]
        marker += 1  # Skip the Add operation (not strictly necessary)
      else:
        layers += [build_affine_matrix(node_proto, None, path[marker-1], reader)]
    elif node_proto.op == 'Relu':
      shape = get_shape(node_proto)
      layers += [{'layer_type': 'relu',
                  'out_depth': shape[0],
                  'out_sx': shape[1],
                  'out_sy': shape[2]}]
    marker += 1

  print 'Done.'
  return {'layers': layers}

parser = argparse.ArgumentParser(description='Convert a TensorFlow model to ' +
  'a JSON string for use with ConvNetJS.')
parser.add_argument('--checkpoint_file', nargs=1,
  help='The checkpoint file containing variable values.')
parser.add_argument('--meta_file', nargs=1,
  help='The meta file containing the structure of the graph.')
parser.add_argument('--output_file', nargs=1,
  help='The output file containing the graph formatted as JSON.',
  default=['model.json'])
parser.add_argument('--input_node', nargs=1,
  help='The name of the node that holds the input value.')
parser.add_argument('--output_node', nargs=1,
  help='The name of the node that holds the output value.')
args = parser.parse_args()

if args.checkpoint_file[0] and args.meta_file[0]:
  model_json = generate_model_json(args)
  f = open(args.output_file[0], 'w')
  f.write(json.dumps(model_json))
