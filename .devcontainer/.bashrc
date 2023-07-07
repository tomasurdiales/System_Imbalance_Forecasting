# ~/.bashrc: executed by bash(1) for non-login shells.

# Note: PS1 and umask are already set in /etc/profile. You should not
# need this unless you want different defaults for root.
# PS1='${debian_chroot:+($debian_chroot)}\h:\w\$ '
# umask 022

# You may uncomment the following lines if you want `ls' to be colorized:
export LS_OPTIONS='--group-directories-first'
# export LS_OPTIONS='--color=auto' # default colors are pretty wrong here(?)
# eval "$(dircolors)"
export QUOTING_STYLE=literal
alias ls='ls $LS_OPTIONS'

# Set up nicer prompt:
export PS1='\e[1m[\e[1;32m\u@codespaces\e[0m|\e[1;34m\w\e[0m\e[1m] \e[0m'

# Set Simplify project on python path:
export PYTHONPATH=/workspaces/System_Imbalance_Forecasting

# To make path output more readable:
function path(){
    old=$IFS
    IFS=:
    printf ${PATH//:/$'\n'}
    IFS=$old
}
