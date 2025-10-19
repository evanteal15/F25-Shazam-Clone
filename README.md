![header](asset/header.png)

# F25-Shazam-Clone

<img src="asset/shazam.png" height=50/>

The goal of this program is to create a scalable Shazam clone that can identify songs its library from audio input.

This project will combine elements of Machine Learning, Signal Analysis, App Development (React + JavaScript), and Backend Development (SQL + Flask).

## Files:

- `dataloader.py` - interface to work with a tracks dataset as a list of dictionaries (`dataloader.load()`)
- `cm_helper.py` - audio preprocessing, STFT computation, test sample creation
- `cm_visualizations.py` - plots spectrograms with peaks
- `DBcontrol.py` - database for managing many hashes (we'll investigate this later)
- `predict_song.py` - creates a `/predict` endpoint for interfacing with music recognition model
- `const_map.py` - constellation mapping
- `hasher.py` - creates hashes for representing pairs of peaks in database
- `search.py` - detailed implementation of audio search
- `test_hash.py` - creates audio fingerprints
- `test_search.py` - sends a request to `/predict` endpoint

## Timeline

Subject to changes.
| Date | Activity |
|-----------|------------------------------|
| Sept 21 | Introduction + Digital Audio 🔊 |
| Sept 28 | Fourier Transforms, Spectrograms 🧮 |
| Oct 5 | ️Constellation Mapping 🔭 |
| - | Fall Break ️🍂 |
| Oct 19 | Audio Search, Expo Intro 🔍 |
| Oct 26 | Buffer Week, MySQL 💽️|
| Nov 2 | Flask endpoint, Expo 🌐 |
| Nov 9 | Putting it all together 🔧|
| Nov 16 | Prepare for final presentations 🎉 |

## Presentation Slides

[Google Drive](https://docs.google.com/presentation/d/1zfACjefKNI2SxUwyjICdXe_Cc1dKNuPlsfJnkOWKs7I/edit?usp=sharing)


## MySQL

```bash
# macos:
brew install mysql
brew services start mysql
mysql_secure_installation
```





```
Run it yourself!

Download requirements from requirements.txt

If you are using wsl:
Run this command in WSL terminal replacing YOU_WSL2_IP with the IP address of your endpoint
netsh interface portproxy add v4tov4 listenaddress=0.0.0.0 listenport=5003 connectaddress=YOUR_WSL2_IP connectport=5003
Also add a windows defender firewall rule at the selected port
netsh advfirewall firewall add rule name="WSL2 5003" dir=in action=allow protocol=TCP localport=5003
You can now interact with the app backend if it is in a wsl virtual environment!
```