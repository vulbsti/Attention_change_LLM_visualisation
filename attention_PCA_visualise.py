import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns
from sklearn.decomposition import PCA
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---------------------------
# 1. Load the Pretrained Model
# ---------------------------
# (Replace with the actual model ID if different.)
model_name = "microsoft/Phi-3.5-vision-instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    trust_remote_code=True,  # to avoid downloading code from Hugging Face
    torch_dtype = torch.float32,
    _attn_implementation='eager',
    output_hidden_states=True,  # to get hidden states from every transformer block
    output_attentions=True      # to get attention weights
)
model.eval()  # set to evaluation mode

# ---------------------------
# 2. Encode the Prompt and Run the Model
# ---------------------------
prompt = "Why does sun rises in the east."
inputs = tokenizer(prompt, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)

# The output contains:
# - hidden_states: a tuple of (embeddings, layer1, layer2, ... layerN)
# - attentions: a tuple of (layer1, layer2, ... layerN)
# Each is (batch_size, seq_len, hidden_dim)
hidden_states = outputs.hidden_states

# ---------------------------
# 3. Prepare a PCA Projection of the Token Embedding Space
# ---------------------------
# We use the model’s input embedding matrix.
# (In many causal LMs the input and output embeddings are tied.)
token_embedding_matrix = model.get_input_embeddings(
).weight.detach().cpu().numpy()  # shape: (vocab_size, hidden_dim)

# Fit PCA to reduce the embedding space to 2 dimensions.
pca = PCA(n_components=2)
token_embeddings_2d = pca.fit_transform(token_embedding_matrix)

# ---------------------------
# 4. Compute the Evolving “Next-Token” Prediction Across Layers
# ---------------------------
# For each transformer layer (skipping the initial embedding layer at index 0),
# we take the hidden state corresponding to the last token in the prompt.
# Then we project that hidden state into vocabulary space (using the LM head) by
# computing its dot-product with all token embeddings.
# Finally, we take the argmax (most likely token) and get its 2D location.
predicted_positions = []  # will hold (x, y) for each layer’s argmax prediction
predicted_tokens = []     # corresponding decoded token string

# Loop over layers 1 ... N (skipping the embedding output at index 0)
for layer_idx, layer_hidden in enumerate(hidden_states[1:], start=1):
    # layer_hidden shape: (batch_size, seq_len, hidden_dim)
    last_token_hidden = layer_hidden[0, -1, :]  # shape: (hidden_dim,)

    # Compute logits by taking the dot-product with the token embedding matrix.
    # (Some models have additional normalization or bias; this is a simple approximation.)
    # shape: (vocab_size,)
    logits = last_token_hidden @ token_embedding_matrix.T

    # Compute probabilities (softmax)
    probs = torch.softmax(torch.tensor(logits), dim=0).numpy()

    # Get the argmax predicted token id
    predicted_id = int(np.argmax(probs))
    predicted_token = tokenizer.decode([predicted_id])
    predicted_tokens.append(predicted_token)

    # Get the 2D position for the predicted token from our PCA-transformed embeddings.
    pos_2d = token_embeddings_2d[predicted_id]
    predicted_positions.append(pos_2d)

print("Layer-by-layer predicted next tokens:")
for i, token in enumerate(predicted_tokens, start=1):
    print(f"  Layer {i:2d}: {token}")

# ---------------------------
# 5. Animate the Trajectory of the Predicted Token in the 2D Embedding Space
# ---------------------------
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)

ax.set_title("Evolution of Next-Token Prediction in Embedding Space")
ax.set_xlabel("PCA Component 1")
ax.set_ylabel("PCA Component 2")

# Plot all vocabulary tokens in light gray for context.
ax.scatter(token_embeddings_2d[:, 0], token_embeddings_2d[:, 1],
           color='lightgray', s=10, alpha=0.5, label="Vocabulary tokens")

# A red dot will show the predicted token at each layer.
sc = ax.scatter([], [], color='red', s=50, label="Predicted token")
layer_text = ax.text(0.02, 0.95, "", transform=ax.transAxes,
                     fontsize=12, color='blue')
token_text = ax.text(0.02, 0.90, "", transform=ax.transAxes,
                     fontsize=12, color='green')

ax.legend()


def init():
    # sc.set_offsets([])
    sc.set_offsets(np.c_[[], []])
    layer_text.set_text("")
    token_text.set_text("")
    return sc, layer_text, token_text


# def update(frame):
#     pos = predicted_positions[frame]
#     sc.set_offsets([pos])
#     layer_text.set_text(f"Layer: {frame+1}")
#     token_text.set_text(f"Predicted token: {predicted_tokens[frame]}")
#     return sc, layer_text, token_text


def update(frame):
    # Ensure pos is a 2D array of shape (1, 2)
    pos = np.array(predicted_positions[frame]).reshape(1, -1)
    sc.set_offsets(pos)
    layer_text.set_text(f"Layer: {frame+1}")
    token_text.set_text(f"Predicted token: {predicted_tokens[frame]}")
    return sc, layer_text, token_text

ani = FuncAnimation(fig, update, frames=len(predicted_positions),
                    init_func=init, interval=1000, blit=True, repeat=False)

plt.show()

# ---------------------------
# 6. Visualize Attention Weights from the Last Transformer Layer
# ---------------------------
# We average the attention weights over all heads for the last layer.
# The attention weights have shape: (batch_size, num_heads, seq_len, seq_len).
# shape: (seq_len, seq_len)
last_layer_attn = outputs.attentions[-1][0].mean(dim=0).detach().cpu().numpy()

# Get the token list for labeling the heatmap.
input_tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

plt.figure(figsize=(8, 6))
sns.heatmap(last_layer_attn, xticklabels=input_tokens, yticklabels=input_tokens,
            cmap="viridis", annot=True, fmt=".2f")
plt.title("Average Self-Attention Weights (Last Layer)")
plt.xlabel("Key Tokens")
plt.ylabel("Query Tokens")
plt.tight_layout()
plt.show()
