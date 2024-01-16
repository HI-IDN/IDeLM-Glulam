#!/bin/bash

# Default values
hours=8
cpus=12
mem=3900
partition=any_cpu
file=data/glulam.csv
depth=115
version=$(git describe --tags --always --abbrev=0 2>/dev/null)
max_sessions=2
runs=10

# Check if the first parameter (depth) is provided
if [ ! -z $1 ]; then
    depth=$1
fi

if [ ! -z $2 ]; then 
    file=$2
    if [ ! -f $file ]; then echo File $file does not exist; exit; fi
fi

# Check if the second parameter (dependency) is provided
if [ ! -z $3 ]; then 
    dependency="#SBATCH --dependency=afterok"
    for task_id in $(seq 1 $runs); do dependency="${dependency}:$3_${task_id}"; done
fi

# Create a job array with a limit of 2 concurrent jobs
job_script="data/slurm/job_script_${depth}.sh"
mkdir -p data/slurm
cat > "$job_script" <<EOF
#!/bin/bash
#SBATCH --job-name=ES_${depth}
#SBATCH --partition=${partition}
#SBATCH --cpus-per-task=${cpus}
#SBATCH --mem-per-cpu=${mem}
#SBATCH --time=${hours}:00:00
#SBATCH --output=data/slurm/%x_%a_%j.out
#SBATCH --error=data/slurm/%x_%a_%j.err
#SBATCH --array=1-${runs}%${max_sessions}
$dependency

# Run the command for this array job
make data/$version/soln_ES_d${depth}_\${SLURM_ARRAY_TASK_ID}.json depth=${depth} run=\${SLURM_ARRAY_TASK_ID} VERSION=${version} FILE=${file}
EOF

# Submit the job array script
sbatch "$job_script"
