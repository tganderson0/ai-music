import glob
import pickle
import numpy
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM, Activation, BatchNormalization as BatchNorm
from keras.utils import np_utils
from music21 import converter, instrument, note, chord
from keras.callbacks import ModelCheckpoint

def train_network():
    notes = read_midi_notes()

    n_vocab = len(set(notes))

    network_input, network_output = prepare_sequences(notes, n_vocab)

    model = create_network(network_input, n_vocab)

    # if input("Have you updated the weights? (y to continue): ") != 'y':
    #     print("Well update them then")
    #     exit()
    #
    # model.load_weights('new_saved_nets/weights-improvement-999-1.0602-bigger.hdf5')

    train(model, network_input, network_output)

def read_midi_notes():
    """ Converts all the data's notes into a format that can be used y the network. Credit to @Skuldur on github """
    """ The dataset on windows has been training on the full test folder, while the linux has been on just the train_set folder (castlevania) """
    notes = []

    for file in glob.glob("nesmdb_midi/nesmdb_midi/new_train_set/*.mid"):
        midi = converter.parse(file)

        try:
            s2 = instrument.partitionByInstrument(midi)
            notes_to_parse = s2.parts[0].recurse()
        except:
            notes_to_parse = midi.flat.notes

        for element in notes_to_parse:
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))

    with open('data/notes', 'wb') as filepath:
        pickle.dump(notes, filepath)

    return notes

def prepare_sequences(notes, n_vocab):
    # Change the sequence length above to modify
    """ Prepares sequences for the network CREDIT TO @Skuldur. Their github will be in README """
    sequence_length = 100
    # Names of pitches
    pitchnames = sorted(set(item for item in notes))

    # Creates dictionary of pitches so you can use the name to get the integer
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    network_input = []
    network_output = []

    # Create input sequences and the corresponding outputs
    for i in range(0, len(notes) - sequence_length):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])

    n_patterns = len(network_input)

    # Reshapes input into format compatible with LSTM layers
    network_input = numpy.reshape(network_input, (n_patterns, sequence_length, 1))
    # Normalize input
    network_input = network_input / float(n_vocab)

    network_output = np_utils.to_categorical(network_output)

    return network_input, network_output

def create_network(network_input, n_vocab):
    model = Sequential()
    model.add(LSTM(
        512,
        input_shape=(network_input.shape[1], network_input.shape[2]),
        return_sequences=True
    ))

    model.add(LSTM(512))
    model.add(BatchNorm())
    model.add(Dropout(0.3))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(BatchNorm())
    model.add(Dropout(0.3))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(BatchNorm())
    model.add(Dropout(0.3))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(BatchNorm())
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    return model

def train(model, network_input, network_output):
    filepath = "epoch1000-FF/weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )
    callbacks_list = [checkpoint]

    model.fit(network_input, network_output, epochs=4000, batch_size=128, callbacks=callbacks_list)


if __name__ == '__main__':
    train_network()

