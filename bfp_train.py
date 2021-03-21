# Chang's code to restore model
# output name in the model graph (may need to check it using tensorboard)
    saver = tf.compat.v1.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)

    with tf.compat.v1.Session() as sess:
        saver.restore(sess, input_checkpoint) # restore the model parameters
        output_graph_def = tf.compat.v1.graph_util.convert_variables_to_constants(  # freeze the parameters
            sess=sess,
            input_graph_def=sess.graph_def,
            output_node_names=output_node_names.split(",")) # seperate multiple output names using ","

        with tf.io.gfile.GFile(output_graph, "wb") as f: # save the model
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node)) # obtain node #
