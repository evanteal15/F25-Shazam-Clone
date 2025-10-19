#!/bin/bash

set -Eeuo pipefall

# Sanity check command line options
usage() {
  echo "Usage: $0 (mac|linux)"
}

if [ $# -ne 1 ]; then
  usage
  exit 1
fi

case $1 in
    "mac")
        brew install mysql
        brew postinstall mysql
        brew services start mysql
        mysql -e "create database shazamesque;"
        mysql shazamesque
    ;;
    "linux")
        sudo apt update
        sudo apt install mysql-server
        sudo service mysql start
        sudo mysql -e "create database shazamesque; create user 'user'@'localhost' identified by ''; grant all privileges on shazamesque.* to 'user'@'localhost';"
    ;;
    *)
        usage
        exit 1
        ;;
esac

# wsl:

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
#
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