#!/bin/bash

# Use custom .bashrc
cp /workspaces/System_Imbalance_Forecasting/.devcontainer/.bashrc /root/

pre-commit install --install-hooks

apt-get install htop -y
apt-get install speedtest-cli -y
apt-get install ncdu -y
