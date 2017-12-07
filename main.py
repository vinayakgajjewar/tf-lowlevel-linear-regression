# Example using the low-level tf.train API
# Trainable linear regression model

import tensorflow as tf

# Model parameters
# Initialize them with arbitrary values
W = tf.Variable([0], dtype=tf.float32)
b = tf.Variable([0], dtype=tf.float32)
# Model input and output
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
linear_model = W * x + b

# Loss calculation
# I actually have no idea how this works yet
loss = tf.reduce_sum(tf.square(linear_model - y))
# Optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

# Training data
x_train = [-2, -1, 0, 1, 2]
y_train = [-3, -1, 1, 3, 5]
# Training loop
init = tf.global_variables_initializer()
sesh = tf.Session()
sesh.run(init) # Reset values
for i in range(1000):
    sesh.run(train, {x: x_train, y: y_train})

# Evaluate training accuracy
curr_W, curr_b, curr_loss = sesh.run([W, b, loss], {x: x_train, y: y_train})
print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))
