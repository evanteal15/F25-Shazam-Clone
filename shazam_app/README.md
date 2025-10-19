![header](../asset/header.png)

<img src="../asset/shazam.png" height=50/>

## Files:

Helper Files:

- `dataloader.py` - interface to work with a tracks dataset as a list of dictionaries (`dataloader.load()`)
- `cm_helper.py` - audio preprocessing, STFT computation, test sample creation
- `cm_visualizations.py` - plots spectrograms with peaks
- `DBcontrol.py` - database for managing many hashes (we'll investigate this later)
- `predict_song.py` - creates a `/predict` endpoint for interfacing with music recognition model

Core Files:

- `const_map.py` - constellation mapping
- `hasher.py` - creates hashes for representing pairs of peaks in database
- `search.py` - detailed implementation of audio search

Testing:

- `test_hash.py` - creates audio fingerprints
- `test_search.py` - sends a request to `/predict` endpoint
