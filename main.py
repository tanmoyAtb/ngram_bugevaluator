from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import json
import argparse
import collections
import math
import os
import random
import sys
import zipfile
import copy
import numpy as np
from six.moves import urllib
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

from tensorflow.contrib.tensorboard.plugins import projector

window_size = 1
iterations = 100001
plot_points = 60
final_embeddings_file = "./final_embedding_100001_w1_python.txt"
reversed_dictionary_file = "./reverse_dictionary_w1_python.txt"
word_embedding_output_file = "tsne_benchmark_100001_w1_python.png"
training_directory = "./data_python/train"

validation_directory_insert = "./data_python/validation/insert"
validation_directory_delete = "./data_python/validation/delete"
validation_directory_swap = "./data_python/validation/swap"

data_index = 0
def gettempdir():
  return "."

def word2vec_basic(log_dir):
  """Example of building, training and visualizing a word2vec model."""
  # Create the directory for TensorBoard variables if there is not.
  if not os.path.exists(log_dir):
    os.makedirs(log_dir)

  train_dir_name = training_directory

  # Read the data into a list of strings.
  def read_data(train_dir_name):
    data = list()
    for filename in os.listdir(train_dir_name):
      with open(os.path.join(train_dir_name, filename)) as f:
        file_data = f.readlines()
      file_data = [x.strip() for x in file_data if x and x not in ["\n"]]
      file_data = [x for x in file_data if x]
      data += file_data
    return data

  vocabulary = read_data(train_dir_name)
  print('Data size', len(vocabulary))

  # Step 2: Build the dictionary

  def build_dataset(words, n_words):
    """Process raw inputs into a dataset."""
    count = collections.Counter(words).most_common(n_words - 1)
    dictionary = {}
    for word, _ in count:
      dictionary[word] = len(dictionary)
    data = []
    unk_count = 0
    for word in words:
      index = dictionary.get(word, 0)
      data.append(index)
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    vocabulary_size = len(count)
    return data, count, dictionary, reversed_dictionary


  # Filling 4 global variables:
  # data - list of codes (integers from 0 to vocabulary_size-1).
  #   This is the original text but words are replaced by their codes
  # count - map of words(strings) to count of occurrences
  # dictionary - map of words(strings) to their codes(integers)
  # reverse_dictionary - maps codes(integers) to words(strings)
  data, count, unused_dictionary, reverse_dictionary = build_dataset(
      vocabulary, len(vocabulary))
  vocabulary_size = len(count)

  del vocabulary  # Hint to reduce memory.
  print('Most common words', count[:5])
  print('Sample data', data[:10], [reverse_dictionary[i] for i in data[:10]])

  # Step 3: Function to generate a training batch for the skip-gram model.
  def generate_batch(batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span)  # pylint: disable=redefined-builtin
    if data_index + span > len(data):
      data_index = 0
    buffer.extend(data[data_index:data_index + span])
    data_index += span
    for i in range(batch_size // num_skips):
      context_words = [w for w in range(span) if w != skip_window]
      words_to_use = random.sample(context_words, num_skips)
      for j, context_word in enumerate(words_to_use):
        batch[i * num_skips + j] = buffer[skip_window]
        labels[i * num_skips + j, 0] = buffer[context_word]
      if data_index == len(data):
        buffer.extend(data[0:span])
        data_index = span
      else:
        buffer.append(data[data_index])
        data_index += 1
    # Backtrack a little bit to avoid skipping words in the end of a batch
    data_index = (data_index + len(data) - span) % len(data)
    return batch, labels

  batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)
  for i in range(8):
    print(batch[i], reverse_dictionary[batch[i]], '->', labels[i, 0],
          reverse_dictionary[labels[i, 0]])

  # Step 4: Build and train a skip-gram model.

  batch_size = 128
  embedding_size = 128  # Dimension of the embedding vector.
  skip_window = window_size  # How many words to consider left and right.
  num_skips = 2  # How many times to reuse an input to generate a label.
  num_sampled = 64  # Number of negative examples to sample.

  valid_size = 16  # Random set of words to evaluate similarity on.
  valid_window = 50  # Only pick dev samples in the head of the distribution.
  valid_examples = np.random.choice(valid_window, valid_size, replace=False)

  graph = tf.Graph()

  with graph.as_default():

    # Input data.
    with tf.name_scope('inputs'):
      train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
      train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
      valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    # Ops and variables pinned to the CPU because of missing GPU implementation
    with tf.device('/cpu:0'):
      # Look up embeddings for inputs.
      with tf.name_scope('embeddings'):
        embeddings = tf.Variable(
            tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
        embed = tf.nn.embedding_lookup(embeddings, train_inputs)

      # Construct the variables for the NCE loss
      with tf.name_scope('weights'):
        nce_weights = tf.Variable(
            tf.truncated_normal([vocabulary_size, embedding_size],
                                stddev=1.0 / math.sqrt(embedding_size)))
      with tf.name_scope('biases'):
        nce_biases = tf.Variable(tf.zeros([vocabulary_size]))


    #   http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/
    with tf.name_scope('loss'):
      loss = tf.reduce_mean(
          tf.nn.nce_loss(
              weights=nce_weights,
              biases=nce_biases,
              labels=train_labels,
              inputs=embed,
              num_sampled=num_sampled,
              num_classes=vocabulary_size))

    # Add the loss value as a scalar to summary.
    tf.summary.scalar('loss', loss)

    # Construct the SGD optimizer using a learning rate of 1.0.
    with tf.name_scope('optimizer'):
      optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

    # Compute the cosine similarity between minibatch examples and all
    # embeddings.
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings,
                                              valid_dataset)
    similarity = tf.matmul(
        valid_embeddings, normalized_embeddings, transpose_b=True)

    # Merge all summaries.
    merged = tf.summary.merge_all()

    # Add variable initializer.
    init = tf.global_variables_initializer()

    # Create a saver.
    saver = tf.train.Saver()

  # Step 5: Begin training.
  num_steps = iterations

  with tf.Session(graph=graph) as session:
    # Open a writer to write summaries.
    writer = tf.summary.FileWriter(log_dir, session.graph)

    # We must initialize all variables before we use them.
    init.run()
    print('Initialized')

    average_loss = 0
    for step in xrange(num_steps):
      batch_inputs, batch_labels = generate_batch(batch_size, num_skips,
                                                  skip_window)
      feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

      # Define metadata variable.
      run_metadata = tf.RunMetadata()

      # We perform one update step by evaluating the optimizer op (including it
      # in the list of returned values for session.run()
      # Also, evaluate the merged op to get all summaries from the returned
      # "summary" variable. Feed metadata variable to session for visualizing
      # the graph in TensorBoard.
      _, summary, loss_val = session.run([optimizer, merged, loss],
                                         feed_dict=feed_dict,
                                         run_metadata=run_metadata)
      average_loss += loss_val

      # Add returned summaries to writer in each step.
      writer.add_summary(summary, step)
      # Add metadata to visualize the graph for the last run.
      if step == (num_steps - 1):
        writer.add_run_metadata(run_metadata, 'step%d' % step)

      if step % 2000 == 0:
        if step > 0:
          average_loss /= 2000
        # The average loss is an estimate of the loss over the last 2000
        # batches.
        print('Average loss at step ', step, ': ', average_loss)
        average_loss = 0

      # Note that this is expensive (~20% slowdown if computed every 500 steps)
      if step % 10000 == 0:
        sim = similarity.eval()
        for i in xrange(valid_size):
          valid_word = reverse_dictionary[valid_examples[i]]
          top_k = 8  # number of nearest neighbors
          nearest = (-sim[i, :]).argsort()[1:top_k + 1]
          log_str = 'Nearest to %s:' % valid_word
          for k in xrange(top_k):
            close_word = reverse_dictionary[nearest[k]]
            log_str = '%s %s,' % (log_str, close_word)
          print(log_str)
    final_embeddings = normalized_embeddings.eval()

    # Write corresponding labels for the embeddings.
    with open(log_dir + '/metadata.tsv', 'w') as f:
      for i in xrange(vocabulary_size):
        f.write(reverse_dictionary[i] + '\n')

    # Save the model for checkpoints.
    saver.save(session, os.path.join(log_dir, 'model.ckpt'))

    # Create a configuration for visualizing embeddings with the labels in
    # TensorBoard.
    config = projector.ProjectorConfig()
    embedding_conf = config.embeddings.add()
    embedding_conf.tensor_name = embeddings.name
    embedding_conf.metadata_path = os.path.join(log_dir, 'metadata.tsv')
    projector.visualize_embeddings(writer, config)

  writer.close()

  return final_embeddings, reverse_dictionary
  # Step 6: Visualize the embeddings.


def plot(final_embeddings, reverse_dictionary, file_name=None):

  def plot_with_labels(low_dim_embs, labels, filename):
    assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
    plt.figure(figsize=(18, 18))  # in inches
    for i, label in enumerate(labels):
      x, y = low_dim_embs[i, :]
      plt.scatter(x, y)
      plt.annotate(
          label,
          xy=(x, y),
          xytext=(5, 2),
          textcoords='offset points',
          ha='right',
          va='bottom')

    plt.savefig(filename)

  try:
    # pylint: disable=g-import-not-at-top
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt

    tsne = TSNE(
        perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
    plot_only = plot_points
    low_dim_embs = tsne.fit_transform(final_embeddings[:plot_only, :])
    labels = [reverse_dictionary[i] for i in xrange(plot_only)]
    plot_with_labels(low_dim_embs, labels, os.path.join(gettempdir(),
                                                        'tsne_benchmark.png' if file_name is None else file_name))

  except ImportError as ex:
    print('Please install sklearn, matplotlib, and scipy to show embeddings.')
    print(ex)

def get_prob_dist(data, final_embeddings, reverse_dictionary):
  data_len = len(data)
  prob = 0
  for i in range(data_len-1):
    line_1 , line_2 = data[i], data[i+1]
    line_1_index = -1
    line_2_index = -1
    for key, val in reverse_dictionary.items():
      if val == line_1:
        line_1_index = key
      if val == line_2:
        line_2_index = key

    if line_1_index == -1 or line_2_index == -1:
      continue
    summation = 0
    for j in range(len(final_embeddings[0])):
      sub_prob = final_embeddings[line_1_index][j]*final_embeddings[line_2_index][j]
      summation += sub_prob
    prob += summation

  return prob

def validate_tests(dir):

  saved_final_embeddings = np.loadtxt(final_embeddings_file)
  saved_reverse_dictionary = {}
  with open(reversed_dictionary_file, "r") as f:
    temp_reverse_dictionary = json.loads(f.readlines()[0])
  for key, value in temp_reverse_dictionary.items():
    saved_reverse_dictionary[int(key)] = value

  test_dir_name = dir
  result_dict = {}
  for each_dirname in os.listdir(test_dir_name):
    if str(each_dirname).startswith("."):
      continue
    result_dict[each_dirname] = dict()

    for each_filename in os.listdir(os.path.join(test_dir_name, each_dirname)):
      if str(each_filename).startswith("."):
        continue
      with open(os.path.join(test_dir_name, each_dirname, each_filename)) as f:
        file_data = f.readlines()
      file_data = [x.strip() for x in file_data if x and x not in ["\n"]]
      file_data = [x for x in file_data if x]
      value = get_prob_dist(file_data, saved_final_embeddings, saved_reverse_dictionary)
      result_dict[each_dirname][each_filename] = value
  final_dict = collections.OrderedDict()
  sorted_dirs = sorted(result_dict.keys())
  for key in sorted_dirs:
    inner_dict = result_dict[key]
    sorted_keys = sorted(inner_dict.keys())
    final_dict[key] = collections.OrderedDict()
    for sorted_key in sorted_keys:
      final_dict[key][sorted_key] = result_dict[key][sorted_key]
  return final_dict


def roundingVals_toTwoDeci(y):
  for d in y:
    for k, v in d.items():
      v = round(v, 2)
      d[k] = v

def plot_histogram (data):
  import matplotlib.pyplot as plt

  # for key, value in data.items():
  #   plt.bar(np.arange(len(value.values())), height= value.values())
  #   plt.xticks(np.arange(len(value.values())), value.keys())
  #   plt.ylabel(key)
  #   plt.show()

  # The data
  for key, value in data.items():
    for k, v in value.items():
      v = round(v, 2)
      value[k] = v

  smallest = data["smallest"]
  grade = data["grade"]
  checksum = data["checksum"]
  median = data["median"]
  syllables = data["syllables"]
  bubble = data["bubble"]

  indices = ["", "smallest", "grade", "checksum", "median","syllables", "bubble"]

  smX = np.arange(5)
  grX = np.arange(5)+10
  ckX = np.arange(5)+20
  mdX = np.arange(5)+30
  syX = np.arange(5)+40
  bbX = np.arange(5)+50

  width = 0.65


  fig = plt.figure()
  ax = fig.add_subplot(111)
  ax.bar(smX, smallest.values(), width, color = 'b')
  ax.bar(grX, grade.values(), width, color = 'b')
  ax.bar(ckX, checksum.values(), width, color = 'b')
  ax.bar(mdX, median.values(), width, color = 'b')
  ax.bar(syX, syllables.values(), width, color = 'b')
  ax.bar(bbX, bubble.values(), width, color = 'b')

  ax.axes.set_xticklabels(indices)
  ax.set_xlabel('Prob of Insertion (first bar is refCode)')
  ax.set_ylabel('Prob')
  plt.show()

def main(unused_argv):
  # Give a folder path as an argument with '--log_dir' to save
  # TensorBoard summaries. Default is a log folder in current directory.

  # data processing and word embedding training

  """
  current_path = os.path.dirname(os.path.realpath(sys.argv[0]))
  
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--log_dir',
      type=str,
      default=os.path.join(current_path, 'log'),
      help='The log directory for TensorBoard summaries.')
  flags, unused_flags = parser.parse_known_args()
  final_embeddings, reverse_dictionary = word2vec_basic(flags.log_dir)
  

  # data processing and training ends

  # save data to files and plot word embedding

  np.savetxt(final_embeddings_file, final_embeddings)
  with open(reversed_dictionary_file, "w") as f:
    f.write(json.dumps(reverse_dictionary))
  saved_final_embeddings = np.loadtxt(final_embeddings_file)
  saved_reverse_dictionary = {}
  with open(reversed_dictionary_file, "r") as f:
    temp_reverse_dictionary = json.loads(f.readlines()[0])
  for key, value in temp_reverse_dictionary.items():
    saved_reverse_dictionary[int(key)] = value
  
  plot(saved_final_embeddings, saved_reverse_dictionary, file_name=word_embedding_output_file)

  """

  # save data and plotting end

  # produce results on mutated codes

  result = validate_tests(validation_directory_swap)
  plot_histogram((result))
  print(result)

  # result production end


if __name__ == '__main__':
  tf.app.run()
