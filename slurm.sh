#!/bin/bash

# Default values
hours=5
cpus=48
mem=3900
partition=48cpu_192mem

depth=115
version=$(git describe --tags --always --abbrev=0 2>/dev/null)
max_sessions=2
runs=10

# Check if the first parameter (number of hours) is provided
if [ ! -z $1 ]; then
    depth=$1
fi

# Check if the second parameter (number of CPUs) is provided
if [ ! -z $2 ]; then
    version=$2
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
#SBATCH --output=data/slurm/%x_%j_%a.out
#SBATCH --error=data/slurm/%x_%j_%a.err
#SBATCH --array=1-${runs}%${max_sessions}

# Run the command for this array job
echo make data/$version/soln_ES_d${depth}_\${SLURM_ARRAY_TASK_ID}.json depth=${depth} run=\${SLURM_ARRAY_TASK_ID} VERSION=${version}
make data/$version/soln_ES_d${depth}_\${SLURM_ARRAY_TASK_ID}.json depth=${depth} run=\${SLURM_ARRAY_TASK_ID} VERSION=${version}
EOF

# Submit the job array script
sbatch "$job_script"
