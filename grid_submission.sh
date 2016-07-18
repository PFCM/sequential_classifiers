#! /bin/sh

# jic
#$ -S /bin/sh

# let me know what's up
#$ -M Paul.Mathews@ecs.vuw.ac.nz
#$ -m be

# bail if any part freaks out
# set -e

# get the environment set up
source $HOME/.bash_profile

echo //PATH//
echo $PATH

if [ -d /local/tmp/mathewpaul1/$JOB_ID.$SGE_TASK_ID ]; then
    cd /local/tmp/mathewpaul1/$JOB_ID.$SGE_TASK_ID
else
    echo "nowhere to go "
    echo "options are: "
    ls -la /local/tmp
    echo "or more specifically: "
    ls -la /local/temp/mathewpaul1
    echo "how do you expect me to work in these conditions?"
    exit 2
fi

# recall
# stdout goes to:
#    scriptname.o$JOB_ID

echo //UNAME//
uname -n
echo //id, groups//
id
groups

echo //where am i//
pwd
ls

echo //where is home?//
echo $HOME

echo //setting up python environment//
$HOME/.pyenv/bin/pyenv local py3-workspace

echo //running job: $SGE_TASK_ID//
# do a thing
python $HOME/COMP489/sequential_classifiers/grid_search.py $SGE_TASK_ID

output_dir=$HOME/COMP489/sequential_classifiers/grid_runs_permute_round2/$SGE_TASK_ID

mkdir -p $output_dir

echo //done, copying output to: $output_dir
cp -r . $output_dir
