#!/bin/bash

# macos:
brew install mysql
brew postinstall mysql
brew services start mysql

# wsl:
# sudo apt update
# sudo apt install mysql-server
# sudo service mysql start
# sudo mysql -e "create database shazamesque; create user 'user'@'localhost' identified by ''; grant all privileges on shazamesque.* to 'user'@'localhost';"

# ----------------------------
# mysql_secure_installation
# ----------------------------
# if brew services start mysql doesn't work
# (or on WSL)
#
# mysql.server start
# ----------------------------
# other commands for debugging
#
# pkill mysqld
# ls /usr/local/var/mysql
# ----------------------------

mysql -e "create database shazamesque;"
mysql shazamesque

# ----------------------------
# for setting a password:
# (empty string for no password)
#
# mysql -e "create user 'username'@'localhost' identified by 'password';"
# mysql -u username -p shazamesque
#
# mysql -e "alter user 'root'@'localhost' identified by 'password';"
# mysql -u root -p shazamesque
# ----------------------------