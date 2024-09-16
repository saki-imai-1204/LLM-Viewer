import csv
from model_analyzer import ModelAnalyzer
# Initialize the ModelAnalyzer
# model = 'microsoft/phi-2'
# model = 'facebook/opt-1.3b'
# model = "openai-community/gpt2-xl"
model = "EleutherAI/pythia-1.4b"
analyzer = ModelAnalyzer(model, "nvidia_A100", None, source='huggingface')
# Define sequence lengths and number of users
seqlens = [32, 64, 128]
users = [8, 16, 32]
# Open the CSV file to write results
part = model.split("/")[1]
with open(f'inference_times_{part}.csv', 'w', newline='') as csvfile:
    fieldnames = ['model', 'sequence_length', 'users', 'inference_time']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    # Write the header
    writer.writeheader()
    # Iterate over sequence lengths and users
    for sequence_length in seqlens:
        for user in users:
            results = analyzer.analyze_generate_task(
                prompt_len=5,
                gen_len=sequence_length,
                batchsize=user,
                w_bit=16,
                a_bit=16,
                kv_bit=16
            )
            inference_time = results["inference_time"] * 1000  # multiply as it's our values are in ms
            # Write the data to the CSV file
            writer.writerow({
                'model': model,
                'sequence_length': sequence_length,
                'users': user,
                'inference_time': inference_time
            })
print(f"Inference times have been written to inference_times_{part}.csv")