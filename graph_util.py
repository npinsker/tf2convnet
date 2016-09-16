import numpy
import tensorflow as tf
import json
import collections

class TFGraph:
  __nodes_by_key = collections.defaultdict()
  __node_forward_map = collections.defaultdict(list)
  def __init__(self, graph_def):
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
