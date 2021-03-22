import tensorflow as tf
from tensorflow.keras.layers.experimental import preprocessing

import numpy as np
import os
import time
from random import randint 

# General parameters
TEXT_SIZE = 100
BATCH_SIZE = 64
BUFFER_SIZE = 10000 # Number of element in ram
RNN_UNITS = 1024
EMBEDDING_DIM = 256
CHKPNT_DIR = "./AI/train_checkpoint_2"
CHKPNT_PFX = os.path.join(CHKPNT_DIR, "ckpt_{epoch}")
EPOCHS = 100

LOAD_CHKPNT = "./AI/train_checkpoint_2/ckpt_48"
IS_JUST_TEST = True



# That's fine since we don't have that much many rules
text = open("./AI/data.txt", 'rb').read().decode(encoding='utf-8')
vocab = sorted(set(text))
vocab_size = len(vocab)


# I just followed and densified the tensorflow tutorial here
# The vectoriations function for the texts
chr_to_id = preprocessing.StringLookup(vocabulary=list(vocab))
id_to_chr = preprocessing.StringLookup(vocabulary=chr_to_id.get_vocabulary(), invert=True)
ids_to_text = lambda x:tf.strings.reduce_join(id_to_chr(x), axis=-1)

# Vectorisation of the text
text_as_ids = chr_to_id(tf.strings.unicode_split(text, 'UTF-8'))
dataset_ids = tf.data.Dataset.from_tensor_slices(text_as_ids)

# Split dataset in TEXT_SIZE+1 chunks (the +1 is used to genrate input / test)
examples_per_epoch = len(text)//(TEXT_SIZE+1)
sequences = dataset_ids.batch(TEXT_SIZE+1, drop_remainder=True)

# The function to generate input / exepted from dataset element
def generate_input_target(phrase):return phrase[:-1],phrase[1:]

# Compute dataset
dataset = sequences.map(generate_input_target)

# Shuffle dataset
dataset = (
    dataset
    .shuffle(BUFFER_SIZE)
    .batch(BATCH_SIZE, drop_remainder=True)
    .prefetch(tf.data.experimental.AUTOTUNE))


# Création du model
class DisruptRuleAI(tf.keras.Model):
    
    def __init__(self, vocab_size, embedding_dim, rnn_units):
        super().__init__(self)
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(rnn_units,return_sequences=True, return_state=True)
        self.dense = tf.keras.layers.Dense(vocab_size)
        self.loss = tf.losses.SparseCategoricalCrossentropy(from_logits=True)

    def call(self, inputs, states=None, return_state=False, training=False):
        x = inputs
        x = self.embedding(x, training=training)
        if states is None:
            states = self.gru.get_initial_state(x)
        x, states = self.gru(x, initial_state=states, training=training)
        x = self.dense(x, training=training)
        if return_state:
            return x, states
        else:
            return x
        
    def testDataset(self,dataset):
        for input_, exepted in dataset.take(1):
            pred = self(input_)
            prediction = tf.squeeze(tf.random.categorical(pred[0], num_samples=1),axis=-1).numpy()
            loss = self.loss(exepted,pred).numpy().mean()
            print("Input           :",ids_to_text(input_[0]).numpy())
            print("Model Prediction:",ids_to_text(prediction).numpy())
            print("Exepted         :",ids_to_text(exepted[0]).numpy())
            print("Mean loss       :",loss)
         


model = DisruptRuleAI(len(chr_to_id.get_vocabulary()),EMBEDDING_DIM,RNN_UNITS)
print("\n\nModel has been loaded !")

model.compile(optimizer='adam', loss=model.loss)
print("\n\nModel has been compiled !")

model.load_weights(LOAD_CHKPNT)

checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=CHKPNT_PFX,save_weights_only=True)
print("\n\nCheckpoint has been loaded !")


# Copy paste from tensorflow
class GeneratorOneStep(tf.keras.Model):
    def __init__(self, model, id_to_chr, chr_to_id, temperature=1.0):
        super().__init__()
        self.temperature=temperature
        self.model = model
        self.id_to_chr = id_to_chr
        self.chr_to_id = chr_to_id

        # Create a mask to prevent "" or "[UNK]" from being generated.
        skip_ids = self.chr_to_id(['','[UNK]'])[:, None]
        sparse_mask = tf.SparseTensor(
            # Put a -inf at each bad index.
            values=[-float('inf')]*len(skip_ids),
            indices = skip_ids,
            # Match the shape to the vocabulary
            dense_shape=[len(chr_to_id.get_vocabulary())]) 
        self.prediction_mask = tf.sparse.to_dense(sparse_mask)

    @tf.function
    def generate_one_step(self, inputs, states=None):
        # Convert strings to token IDs.
        input_chars = tf.strings.unicode_split(inputs, 'UTF-8')
        input_ids = self.chr_to_id(input_chars).to_tensor()

        # Run the model.
        # predicted_logits.shape is [batch, char, next_char_logits] 
        predicted_logits, states =  self.model(inputs=input_ids, states=states, 
                                            return_state=True)
        # Only use the last prediction.
        predicted_logits = predicted_logits[:, -1, :]
        predicted_logits = predicted_logits/self.temperature
        # Apply the prediction mask: prevent "" or "[UNK]" from being generated.
        predicted_logits = predicted_logits + self.prediction_mask

        # Sample the output logits to generate token IDs.
        predicted_ids = tf.random.categorical(predicted_logits, num_samples=1)
        predicted_ids = tf.squeeze(predicted_ids, axis=-1)

        # Convert from token ids to characters
        predicted_chars = self.id_to_chr(predicted_ids)

        # Return the characters and model state.
        return predicted_chars, states

    def one_rule(self,length=255,base = None):
        if base == None:base = str(randint(1,9999))+")\r\n"
        next_char = tf.constant([base for _ in range(5)])
        states = None
        result = [next_char]
        
        for n in range(length):
            next_char, states = self.generate_one_step(next_char, states=states)
            result.append(next_char)
            
        result = tf.strings.join(result)
        return result[0]
        


model_generator = GeneratorOneStep(model, id_to_chr, chr_to_id)

print("\n\nGenerator has been loaded !")