#!/bin/bash

# Use custom .bashrc
cp /workspaces/Simplify_MScThesis_Tomas/.devcontainer/.bashrc /root/

pre-commit install --install-hooks

apt-get install htop -y
apt-get install speedtest-cli -y
apt-get install ncdu -y
