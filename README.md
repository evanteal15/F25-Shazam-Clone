# MDST F25-Shazam-Clone

The goal of this program is to create a scalable Shazam clone that can identify songs its library from audio input.

This project will combine elements of Machine Learning, Signal Analysis, App Development (React + JavaScript), and Backend Development (SQl + Falsk).

# Week 1:

1.

# Week 2:

1.

# Week 3:

1.

# Week 4:

1.

# Week 5:

1.

# Week 6:

1.

# Week 7:

1.

# Week 8:

1.

Run it yourself!

Download requirements from requirements.txt

If you are using wsl:
Run this command in WSL terminal replacing YOU_WSL2_IP with the IP address of your endpoint
netsh interface portproxy add v4tov4 listenaddress=0.0.0.0 listenport=5003 connectaddress=YOUR_WSL2_IP connectport=5003
Also add a windows defender firewall rule at the selected port
netsh advfirewall firewall add rule name="WSL2 5003" dir=in action=allow protocol=TCP localport=5003
You can now interact with the app backend if it is in a wsl virtual environment!
