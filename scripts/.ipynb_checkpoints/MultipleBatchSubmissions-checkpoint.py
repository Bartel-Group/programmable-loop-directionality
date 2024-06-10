import csv
import subprocess
import os

def submit_multiple_slurm(folder_path):
    
    files = os.listdir(folder_path)
    
    for file in files:
        file_path = os.path.join(folder_path, file)
        if os.path.isfile(file_path):
            
            with open(file_path, 'r') as f:
                csv_reader = csv.reader(f)
                header = next(csv_reader) # skips the header line
                batch_number = next(csv_reader)[0]
                
            slurm_cmd = f"sbatch --job-name=RR11A_{batch_number}  --output=RR11A_{batch_number}.out --export=ALL,file_path={file_path}  DynamicLoop-Parallel-1.slurm"
            subprocess.run(slurm_cmd, shell=True)
    
path = "/home/dauenha0/murp1677/Cyclic_Dynamics/Batch_Scripts/Parameters/ReRuns_Round11_Set1"
submit_multiple_slurm(path)

