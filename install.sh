export TMPDIR=$HOME/tmp
export TEMP=$HOME/tmp
export TMP=$HOME/tmp
mkdir -p $HOME/tmp
mkdir -p $HOME/.cache/pip
export XDG_CACHE_HOME=$HOME/.cache
echo 'export PATH=$PATH:$HOME/.local/bin' >> ~/.bashrc
source ~/.bashrc