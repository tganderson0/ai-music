# AI Music Creation - (Project 2 for CS5600)

## Overview

For my project, I was inspired by an earlier reading where the authors used a neural network to generate
artwork. I wanted to do something similar, so I decided to create a music generating network.
My original plan was to have a network that produced an entire song at a time, but instead, I made
one that would predict note by note, using the previous notes to predict. Much of the work for the MIDI
side of the project was taken from https://github.com/Skuldur/Classical-Piano-Composer. His work allowed me to
work on my network itself, without having to study a bunch of music theory and how MIDI files are structured.

## Libaries required

* Music21 (used for all the MIDI operations)
* Keras (network)
* h5py (for the weights)
* numpy (network)
* pickle (saving the 'notes' for generation)
* glob (reading the MIDIs)


## Running the project

NOTE:

If you would like to train with new songs, make sure that you rename the 'data/notes' file, as that one is generated based on the
songs that are trained on. It represents all the potential options that the network can choose.

To train the network, just run music_network.py. The songs it will currently take from are in 'nesmdb_midi/new_train_set'.
It is a collection of Final Fantasy I music.

```shell script
$ python3 music_network.py
```

To generate music, run generate_music.py. It will output 5 midi files each run, named 'test_output_N.midi', where N is the song
number.

```shell script
$ python3 generate_music.py
```

The project's output will be a MIDI file, which I used https://onlinesequencer.net/import to convert the MIDI into an MP3
for me. I spent some time trying to get Python itself to play the MIDI for me, but I was not able to get it working without
a large amount of required installations. In the interest of keeping this project's dependencies to a minimum,
I decided to just use the online converter. I have included a few example outputs as well, that are able to be just listened to. 

## Notes and Observations

When I first started this project, I started out with a larger dataset, which was all of the Castlevania music from games 1-4.
The dataset ran extremely slow, which was surprising. I trained for about 1000 epochs, and when listening to the music that it created,
it sounded awful. After some testing, I noticed that my possible notes were ~17,000 options! This was leading
to the extremely slow training time, as well as sounding awful, since almost any option works when you have that many outputs.
To mitigate this, I switched to a different game's music, one with less range in the notes,
and put less overall songs. This almost halved the training time, with results that I would even go as far as to say sound good!

I did notice that many songs it would generate would end up with the same motif put into each of the songs, which was 
very cool for creating songs with similar themes, although less so if you wanted more varied music.

### What did I learn?

One of the first things I learned was the importance of a good data-set. With project 1, we had the advantage
of having a great data-set to use, which made it so the only thing we really had to change was the network.
With this project, I ran into the issue of my data-set not being the best, and it was actually the data-set itself
that I had to change to make it produce good songs, rather than the network structure. I had originally trained with a smaller
network, but changing that did not change the output really at all, which was when I started changing the data-set.

By far, the most important thing I have learned from this, is the importance of good data. With brute forcing it,
the data itself is integral to good results, even before any training has happened. Along those lines, making sure that the
data-set itself works well is a pain. Many of the songs in the data-set I got use different MIDI formats, specifically with marking
keys and changing the key in the middle of the song.

### Future work on project

I really enjoyed this project, and it is something that I would like to continue working after the semester ends. I want to add a network that can create the rest of the instrument tracks, such as the drums, so that it creates
a full fledged song, rather than just the melody. Additionally, I want to train on more varied music, to get some
 different songs.

### Quick Links

All my pre-generated output is saved in the folder [saved-music](./saved-music), which you can just open and listen to.

Network training is contained in [music_network.py](./music_network.py)

Music production is contained in [generate_music.py](./generate_music.py)

The songs that I used to train in the final iteration are contained in the folder [new_train_set](./nesmdb_midi/nesmdb_midi/new_train_set)


