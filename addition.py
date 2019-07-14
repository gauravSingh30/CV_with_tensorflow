import tensorflow as tf
g = tf.Graph()
with g.as_default():
  x = tf.constant(8, name="x")
  y = tf.constant(10, name="y")
  z = tf.constant(11, name="z")
  mysum = tf.add(x, y, name="sum")
  final = tf.add(mysum, z, name="final")

  with tf.Session() as sess:
    print(final.eval())