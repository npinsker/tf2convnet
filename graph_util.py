import numpy
import tensorflow as tf
import json
import collections

class TFGraph:
  __nodes_by_key = collections.defaultdict()
  __node_forward_map = collections.defaultdict(list)
  __graph_def = None
  def __init__(self, graph_def):
    self.__graph_def = graph_def
    for node in graph_def.node:
      self.__nodes_by_key[node.name] = node
      for prev_node_name in node.input:
        self.__node_forward_map[prev_node_name] += [node.name]

  def get_node(self, node_name):
    return self.__nodes_by_key[node_name]

  # Returns a list of nodes that this node depends on.
  def get_previous(self, node_name):
    return self.__nodes_by_key[node_name].input

  # Returns a list of nodes that depend on this node.
  def get_next(self, node_name):
    return self.__node_forward_map[node_name]

  # Returns a list of all nodes.
  def get_all_nodes(self):
    return self.__graph_def.node

def find_longest_path(graph):
  SUPPORTED_NODE_TYPES = ['Placeholder', 'MatMul', 'Add', 'Relu']
  SOURCE_NODE_TYPE = 'Placeholder'
  NEGINF = -100000
  length_to_start = {}

  # TODO: check for infinite loops and terminate cleanly
  def find_length_to_start(node_name, length_to_start):
    if node_name in length_to_start:
      return length_to_start[node_name]

    node = graph.get_node(node_name)
    if node.op == SOURCE_NODE_TYPE and len(node.input) == 0:
      length_to_start[node_name] = 0
      return 0
    elif node.op not in SUPPORTED_NODE_TYPES:
      return NEGINF
    length_to_start[node_name] = 1 + max(
      [find_length_to_start(prev_node_name, length_to_start) \
        for prev_node_name in graph.get_previous(node_name)])
    return length_to_start[node_name]
  
  best_node = None
  for end_node in graph.get_all_nodes():
    if not best_node or find_length_to_start(end_node.name, length_to_start) > \
                        find_length_to_start(best_node.name, length_to_start):
      best_node = end_node
  path = [best_node.name]
  marker = best_node.name
  while length_to_start[marker] != 0:
    for prev_node_name in graph.get_previous(marker):
      if prev_node_name in length_to_start and \
         length_to_start[prev_node_name] == length_to_start[marker]-1:
        path = [prev_node_name] + path
        marker = prev_node_name
        break
  return path
