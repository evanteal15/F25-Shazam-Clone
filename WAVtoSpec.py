import librosa
import hasher

Fs, song = librosa.load('', sr=None)

constellation_map = hasher.create_constellation_map(song, Fs)
hashed_constellation = hasher.hash_constellation(constellation_map)


