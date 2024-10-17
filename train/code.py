import numpy as np
import librosa
import tensorflow.compat.v1 as tf
import tf_slim.layers as _layers
import os
import matplotlib.pyplot as plt

def audio_read(f):
    y, sr = librosa.core.load(f, sr=22050)
    d = librosa.core.get_duration(y=y, sr=sr)
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=128)
    S = np.transpose(np.log(1+10000*S))
    S = np.expand_dims(S, axis=0)
    return y, S, int(d)

def chunk(incoming, n_chunk):
    input_length = incoming.shape[1]
    chunk_length = input_length // n_chunk
    outputs = []
    for i in range(incoming.shape[0]):
        for j in range(n_chunk):
            outputs.append(incoming[i, j*chunk_length:(j+1)*chunk_length, :])
    outputs = np.array(outputs)
    return outputs

def positional_encoding(batch_size, n_pos, d_pos):
    # keep dim 0 for padding token position encoding zero vector
    position_enc = np.array([
        [pos / np.power(10000, 2 * (j // 2) / d_pos) for j in range(d_pos)]
        if pos != 0 else np.zeros(d_pos) for pos in range(n_pos)])

    position_enc[1:, 0::2] = np.sin(position_enc[1:, 0::2]) # dim 2i
    position_enc[1:, 1::2] = np.cos(position_enc[1:, 1::2]) # dim 2i+1
    position_enc = np.tile(position_enc, [batch_size, 1, 1])
    return position_enc

class MusicHighlighter(object):
    def __init__(self):
        self.dim_feature = 64

        self.bn_params = {'is_training': True,
                        'center': True, 'scale': True,
                        'updates_collections': None}
        
        self.x = tf.placeholder(tf.float32, shape=[None, None, 128])
        self.y = tf.placeholder(tf.float32, shape=[None, 174])
        self.pos_enc = tf.placeholder(tf.float32, shape=[None, None, self.dim_feature*4])
        self.num_chunk = tf.placeholder(tf.int32)
        self.build_model()
        self.saver = tf.train.Saver(tf.global_variables())
    
    def conv(self, inputs, filters, kernel, stride):
        dim = inputs.get_shape().as_list()[-2]
        return _layers.conv2d(inputs, filters, 
                                        [kernel, dim], [stride, dim],
                                        activation_fn=tf.nn.relu,
                                        normalizer_fn=_layers.batch_norm,
                                        normalizer_params=self.bn_params)

    def fc(self, inputs, num_units, act=tf.nn.relu):
        return _layers.fully_connected(inputs, num_units,
                                                 activation_fn=act,
                                                 normalizer_fn=_layers.batch_norm,
                                                 normalizer_params=self.bn_params)

    def attention(self, inputs, dim):
        outputs = self.fc(inputs, dim, act=tf.nn.tanh)
        outputs = self.fc(outputs, 1, act=None)
        attn_score = tf.nn.softmax(outputs, dim=1)
        return attn_score

    def build_model(self):
        net = tf.expand_dims(self.x, axis=3)
        net = self.conv(net, self.dim_feature, 3, 2)
        net = self.conv(net, self.dim_feature*2, 4, 2)
        net = self.conv(net, self.dim_feature*4, 4, 2)

        net = tf.squeeze(tf.reduce_max(net, axis=1), axis=1)
        
        net = tf.reshape(net, [1, self.num_chunk, self.dim_feature*4])

        attn_net = net + self.pos_enc
        attn_net = self.fc(attn_net, self.dim_feature*4)
        attn_net = self.fc(attn_net, self.dim_feature*4)
        self.attn_score = self.attention(attn_net, self.dim_feature*4)

        net = self.fc(net, 1024)
        chunk_predictions = self.fc(net, 174, act=tf.nn.softmax)
        self.overall_predictions = tf.squeeze(tf.matmul(self.attn_score, chunk_predictions, transpose_a=True), axis=1)

        # overall_predictions = tf.clip_by_value(overall_predictions, 1e-10, 1.0)

        self.loss = tf.reduce_mean(-tf.reduce_sum(self.y * tf.log(self.overall_predictions), axis=1))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.loss)


def load_song_names(file_path, base_directory):
    with open(file_path, 'r') as f:
        song_names = [line.strip() for line in f.readlines()]
    
    song_paths = [os.path.join(base_directory, name + '.mp3') for name in song_names]
    
    return song_paths

def load_hard_annotations(file_path):
    return np.loadtxt(file_path, delimiter=',')

def train_model(model, song_paths, labels, epochs=5):

    loss_values = []  
    accuracies = []   

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver()

        for epoch in range(epochs):
            epoch_loss = 0
            correct_predictions = 0

            for i in range(len(song_paths)):
                song_path = song_paths[i]
                label = labels[i]

                audio, spectrogram, duration = audio_read(song_path)

                n_chunk, remainder = np.divmod(duration, 3)

                chunk_spec = chunk(spectrogram, n_chunk)

                pos = positional_encoding(batch_size=1, n_pos=n_chunk, d_pos=model.dim_feature * 4)

                n_chunk = n_chunk.astype('int')
                chunk_spec = chunk_spec.astype('float')
                pos = pos.astype('float')

                # Huấn luyện mô hình với bài hát hiện tại
                feed_dict = {
                    model.x: chunk_spec,
                    model.y: label,
                    model.pos_enc: pos,
                    model.num_chunk: n_chunk
                }

                _, loss_value = sess.run([model.optimizer, model.loss], feed_dict=feed_dict)
                epoch_loss += loss_value

                predicted_probabilities = sess.run(tf.nn.softmax(model.overall_predictions), feed_dict=feed_dict)

                predicted_index = np.argmax(predicted_probabilities, axis=1)[0] 

                if label[0, predicted_index] == 1: 
                    correct_predictions += 1

            avg_loss = epoch_loss / len(song_paths)
            accuracy = correct_predictions / len(song_paths)
            loss_values.append(avg_loss)
            accuracies.append(accuracy)
            
            print(f'Epoch {epoch + 1}, Average Loss: {avg_loss}, Accuracy: {accuracy}')
        model.saver.save(sess, 'model/model')
        print("Save success")

    # Trực quan hóa loss và accuracy
    plt.figure(figsize=(12, 5))

    # Đồ thị loss
    plt.subplot(1, 2, 1)
    plt.plot(loss_values, label='Loss', color='blue')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Đồ thị accuracy
    plt.subplot(1, 2, 2)
    plt.plot(accuracies, label='Accuracy', color='green')
    plt.title('Accuracy Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    base_directory = 'CAL500_32kps'
    
    song_paths = load_song_names('songNames.txt', base_directory)
   
    labels = load_hard_annotations('hardAnnotations.txt')

    model = MusicHighlighter()

    train_model(model, song_paths, labels, epochs=1)

