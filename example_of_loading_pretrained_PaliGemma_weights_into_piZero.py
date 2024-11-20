import torch
import time
from pi_zero_pytorch import PiZero
from transformers import PaliGemmaForConditionalGeneration

# Step 1: Load the Pretrained PaliGemma Model (requires pip install transformers huggingface_hub)
# PaliGemma is a pretrained model designed for vision and language tasks.
model_id = "google/paligemma-3b-pt-224"
paligemma_model = PaliGemmaForConditionalGeneration.from_pretrained(model_id)

# Step 2: Initialize the π₀ Model
# π₀ is the custom robotic foundation model with configurable input dimensions.
# Ensure that the `dim` parameter matches the embedding dimension of PaliGemma.
model = PiZero(
    dim=512,  # Embedding dimension for vision and language
    dim_action_input=6,  # Action input dimension
    dim_joint_state=12,  # Joint state input dimension
    num_tokens=20_000    # Vocabulary size
)

# Step 3: Transfer Weights from PaliGemma to π₀ Model
# Extract the state dictionaries from both models.
paligemma_state_dict = paligemma_model.state_dict()
pi_zero_state_dict = model.state_dict()

# Map weights from PaliGemma to π₀ model. This example assumes a direct mapping
# for simplicity. You may need to customize this based on layer compatibility.
for name, param in paligemma_state_dict.items():
    if name in pi_zero_state_dict:
        pi_zero_state_dict[name].data.copy_(param.data)

# Load the updated state dictionary into the π₀ model.
model.load_state_dict(pi_zero_state_dict)

# Step 4: Save the Updated π₀ Model
# Save the model's state dictionary for future use.
model_save_path = 'pi_zero_with_paligemma_weights.pth'
torch.save(model.state_dict(), model_save_path)
print(f"Weights saved as {model_save_path}")

# Set the model to evaluation mode (optional but recommended for testing or inference).
model.eval()

# Step 5: Test the Updated π₀ Model
# Prepare dummy inputs for vision, commands, joint state, and actions.
vision = torch.randn(1, 1024, 512)  # Batch size of 1, sequence length of 1024, embedding size of 512
commands = torch.randint(0, 20_000, (1, 1024))  # Batch size of 1, sequence length of 1024
joint_state = torch.randn(1, 12)  # Batch size of 1, joint state dimension of 12
actions = torch.randn(1, 32, 6)  # Batch size of 1, trajectory length of 32, action dimension of 6

# Perform a forward pass to compute the loss.
start = time.time()
loss, _ = model(vision, commands, joint_state, actions)
loss.backward()  # Backpropagation (useful during training)

# After training, generate sampled actions from the model.
with torch.no_grad():  # Disable gradient computation for inference
    sampled_actions = model(vision, commands, joint_state, trajectory_length=32)  # (1, 32, 6)

end = time.time()

# Display the results
print(f"Sampled actions: {sampled_actions}")
print(f"Type of sampled actions: {type(sampled_actions)}")
print(f"Shape of sampled actions: {sampled_actions.shape}")
print(f"Time taken: {round(end - start)} seconds")
