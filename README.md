# tf2convnet
### TensorFlow to ConvNetJS model converter

This is a small program for converting simple TensorFlow models into JSON loadable by ConvNetJS. It was made so that models trained using TensorFlow can be easily converted into JavaScript settings and used on the web.

Currently, this program only supports fully-connected layers and ReLU layers (in other words, multi-layered perceptrons).

#### How to use the converter

Saving a TensorFlow model with `tf.train.Saver` should produce two files: a <i>checkpoint file</i> (postfaced with `.ckpt`) and a <i>meta graph file</i> (postfaced with `.ckpt.meta`). The checkpoint file contains information about the values of Variables, and the meta graph file contains information about the structure of the model's computation graph.

With these two files, you can run the converter:

`python main.py --checkpoint_file model.ckpt --meta_file model.ckpt.meta --output_file model.json`

The program will try to autodetect the most appropriate nodes to use as the input and output of the model. (Specifically, it will look for the longest path it can find between a 'Placeholder' type node and ends at a node type that it knows how to process.) It will print out some information about the path it found. Give it a glance and make sure it seems right.

If autodetection isn't working, then you need to specify the input nodes manually. You'll need to tell TensorFlow to associate your input and output nodes with specific strings, so that tf2convnet can locate these nodes in the graph.

In your model, if `x` is your input and `y` is your output, then add to your model:

```
tf.add_to_collection('input_node', x)
tf.add_to_collection('output_node', y)

tf.train.export_meta_graph(
    filename='model.ckpt.meta',
    collection_list=['input_node', 'output_node'])
```

And that's all! Alternatively, if you know the name of the input and output nodes in the meta graph file, you can specify them on the command line:

`python main.py --checkpoint_file model.ckpt --meta_file model.ckpt.meta --output_file model.json --input_node INPUT --output_node OUTPUT`
