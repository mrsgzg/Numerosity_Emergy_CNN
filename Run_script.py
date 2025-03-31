import subprocess

def run_commands(commands):
    for cmd in commands:
        print(f"Running: {cmd}")
        process = subprocess.run(cmd, shell=True, text=True, check=True)
        print("Output:", process.stdout)
        print("Error:", process.stderr)
        if process.returncode != 0:
            print("Command failed with return code", process.returncode)
            break

if __name__ == "__main__":
    commands = [
        "python numberosity-neuron-analysis-2.py --model_dir /home/embody_data/Numerosity_Emergy_CNN/activation_data/CNN_3_seed21_kaiming_normal_20250328_161548/CNN_3 --layer block3",
        "python numberosity-neuron-analysis-2.py --model_dir /home/embody_data/Numerosity_Emergy_CNN/activation_data/CNN_1_seed21_kaiming_normal_20250328_153931/CNN_1 --layer block1",
        "python numberosity-neuron-analysis-2.py --model_dir /home/embody_data/Numerosity_Emergy_CNN/activation_data/CNN_4_seed21_kaiming_normal_20250328_172230/CNN_4 --layer block4"
    ]
    
    run_commands(commands)
