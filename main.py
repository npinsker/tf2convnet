import numpy as np
import numpy.fft as fft
import tensorflow as tf
import random
import math

def weight_variable(shape):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial)

def bias_variable(shape):
  initial = tf.constant(0.01, shape=shape)
  return tf.Variable(initial)

SAMPLE_SIZE = 2

x = tf.placeholder(tf.float32, [None, SAMPLE_SIZE])

LAYER1_SIZE = 1
OUTPUT_SIZE = 2

W1 = weight_variable([SAMPLE_SIZE, OUTPUT_SIZE])
b1 = tf.Variable([333,444], dtype='float32')

y = tf.nn.relu(tf.matmul(x, W1) + b1)

y_ = tf.placeholder(tf.float32, [None, OUTPUT_SIZE])
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(y, y_)

#train_step = tf.train.AdamOptimizer(4e-4).minimize(cross_entropy)

#correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
#accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

INPUT_NODE_NAME = 'Placeholder'
OUTPUT_NODE_NAME = 'Relu'

saver = tf.train.Saver()

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

graph_def = tf.get_default_graph().as_graph_def(add_shapes=True)

nodes_by_key = {}

for node in graph_def.node:
  nodes_by_key[node.name] = node

def find_node_path(input_node, output_node, nodes_by_key):
  if input_node not in nodes_by_key:
    raise ValueError('Input node %s not in node dictionary.' % input_node)
  if output_node not in nodes_by_key:
    raise ValueError('Output node %s not in node dictionary.' % output_node)

  def search_for_node(cur_node_proto, target_node_name, nodes_by_key):
    # We search backwards through the operation graph to find the input node.
    if cur_node_proto.name == target_node_name:
      return [target_node_name]
    for prev_node in cur_node_proto.input:
      path = search_for_node(nodes_by_key[prev_node], target_node_name, nodes_by_key)
      if path:
        return path + [cur_node_proto.name]

  return search_for_node(nodes_by_key[output_node], input_node, nodes_by_key)

reader = tf.train.NewCheckpointReader('model.ckpt')
path = find_node_path(INPUT_NODE_NAME, OUTPUT_NODE_NAME, nodes_by_key)

layers = []
marker = 0

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
    json_arr['w'][str(i)] = arr[i]
  return json_arr

def build_affine_matrix(matrix_node, add_node, prev_node):
  variable_node = matrix_node.input[0]
  if variable_node == prev_node:
    variable_node = matrix_node.input[1]
  if variable_node.endswith('/read'):
    variable_node = variable_node[:-5]

  variable_tensor = reader.get_tensor(variable_node)
  bias_tensor = None
  for prev_node in add_node.input:
    if prev_node != matrix_node.name:
      if prev_node.endswith('/read'):
        prev_node = prev_node[:-5]
      bias_tensor = reader.get_tensor(prev_node)
      break
  fc_array = {'layer_type': 'fc',
               'out_depth': 1,
               'out_sx': 1,
               'out_sy': 1,
               'biases': array_to_json(bias_tensor),
               'filters': {}}
  for col_index in range(variable_tensor.shape[1]):
    fc_array['filters'][str(col_index)] = array_to_json(variable_tensor[:,col_index])

  return fc_array

while marker < len(path):
  node_proto = nodes_by_key[path[marker]]
  if node_proto.op == 'Placeholder':
    shape = get_shape(node_proto)
    layers += [{'layer_type': 'input',
                'out_depth': shape[0],
                'out_sx': shape[1],
                'out_sy': shape[2]}]
  elif node_proto.op == 'MatMul':
    if marker < len(path) - 1 and nodes_by_key[path[marker+1]].op == 'Add':
      layers += [build_affine_matrix(node_proto, nodes_by_key[path[marker+1]],
                                    nodes_by_key[path[marker-1]].name)]
  elif node_proto.op == 'Relu':
    shape = get_shape(node_proto)
    layers += [{'layer_type': 'relu',
                'out_depth': shape[0],
                'out_sx': shape[1],
                'out_sy': shape[2]}]
  marker += 1

print layers
