import os
import torch
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from sklearn.metrics.pairwise import cosine_similarity
from transformers import get_scheduler
from sentence_transformers import SentenceTransformer
import evaluate
from huggingface_hub import login

hf_token = os.getenv("HF_TOKEN")
login(hf_token)

# Step 1: RL Environment Setup

class RAGEnvironment:
    def __init__(self, model_path, tokenizer_path, embedding_model_path="sentence-transformers/all-mpnet-base-v2"):
        self.model = AutoModelForCausalLM.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.embedding_model = SentenceTransformer(embedding_model_path)

        # Set up evaluation metrics
        self.rouge = evaluate.load('rouge')
        self.bleu = evaluate.load('bleu')

    def generate_response(self, query, retrieved_docs, max_length=512):
        """Generate response using the current model state"""
        context = f"Query: {query}\n\nContext:\n" + "\n".join([f"- {doc}" for doc in retrieved_docs])
        prompt = f"{context}\n\nBased on the above information, please answer the query concisely and accurately."

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id if self.tokenizer.pad_token_id else self.tokenizer.eos_token_id
            )

        response = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        # Extract only the generated answer part
        response = response.split("Based on the above information, please answer the query concisely and accurately.")[-1].strip()
        return response

# Step 2: Reward Model Development

class RewardModel:
    def __init__(self, embedding_model_path="sentence-transformers/all-mpnet-base-v2"):
        self.embedding_model = SentenceTransformer(embedding_model_path)
        # Define weights for different reward components
        self.weights = {
            "factual_accuracy": 0.5,
            "relevance": 0.3,
            "conciseness": 0.2
        }

    def calculate_factual_accuracy(self, response, retrieved_docs):
        """Calculate how factually accurate the response is compared to retrieved documents"""
        # Embed the response and each document
        response_embedding = self.embedding_model.encode(response)
        doc_embeddings = [self.embedding_model.encode(doc) for doc in retrieved_docs]

        # Calculate max similarity between response and any document
        similarities = [cosine_similarity([response_embedding], [doc_emb])[0][0] for doc_emb in doc_embeddings]
        return max(similarities) if similarities else 0.0

    def calculate_relevance(self, response, query):
        """Calculate how relevant the response is to the original query"""
        response_embedding = self.embedding_model.encode(response)
        query_embedding = self.embedding_model.encode(query)

        similarity = cosine_similarity([response_embedding], [query_embedding])[0][0]
        return similarity

    def calculate_conciseness(self, response):
        """Calculate conciseness score based on response length"""
        # Penalize very short (<20 words) or very long (>200 words) responses
        word_count = len(response.split())

        if word_count < 20:
            return max(0.5, word_count / 20)  # Gradually increase from 0.5 to 1.0
        elif word_count <= 150:
            return 1.0  # Ideal range
        else:
            return max(0.0, 1.0 - (word_count - 150) / 150)  # Gradually decrease to 0

    def calculate_reward(self, response, query, retrieved_docs, ground_truth=None):
        """Calculate the overall reward for a response"""
        factual_score = self.calculate_factual_accuracy(response, retrieved_docs)
        relevance_score = self.calculate_relevance(response, query)
        conciseness_score = self.calculate_conciseness(response)

        # Optional: If ground truth is available, calculate similarity to it
        ground_truth_score = 0.0
        if ground_truth:
            response_embedding = self.embedding_model.encode(response)
            ground_truth_embedding = self.embedding_model.encode(ground_truth)
            ground_truth_score = cosine_similarity([response_embedding], [ground_truth_embedding])[0][0]

        # Calculate weighted sum of reward components
        reward = (
            self.weights["factual_accuracy"] * factual_score +
            self.weights["relevance"] * relevance_score +
            self.weights["conciseness"] * conciseness_score
        )

        # Include ground truth in evaluation if available
        if ground_truth:
            reward = 0.7 * reward + 0.3 * ground_truth_score

        return reward, {
            "factual_accuracy": factual_score,
            "relevance": relevance_score,
            "conciseness": conciseness_score,
            "ground_truth_similarity": ground_truth_score if ground_truth else None,
            "total_reward": reward
        }

# Step 3: PPO Implementation

class PPOTrainer:
    def __init__(
        self,
        model_path,
        tokenizer_path,
        lr=5e-5,
        gamma=0.99,
        clip_range=0.2,
        value_coef=0.5,
        entropy_coef=0.01,
        max_grad_norm=0.5
    ):
        # Policy model (will be updated)
        self.policy = AutoModelForCausalLM.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        # Add padding token if it doesn't exist
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.policy.config.pad_token_id = self.policy.config.eos_token_id

        # Value network for estimating advantage
        # Using a separate language model head for value prediction
        self.value_head = torch.nn.Linear(self.policy.config.hidden_size, 1)

        # Set up optimizer
        self.optimizer = torch.optim.AdamW([
            {'params': self.policy.parameters(), 'lr': lr},
            {'params': self.value_head.parameters(), 'lr': lr}
        ])

        # PPO hyperparameters
        self.gamma = gamma
        self.clip_range = clip_range
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

        # Move models to device (GPU if available)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy.to(self.device)
        self.value_head.to(self.device)

        # Reference model (for calculating KL divergence)
        self.ref_model = AutoModelForCausalLM.from_pretrained(model_path)
        self.ref_model.to(self.device)
        for param in self.ref_model.parameters():
            param.requires_grad = False

    def compute_advantages(self, rewards, values, dones, next_value=0):
        """Compute advantages using Generalized Advantage Estimation (GAE)"""
        advantages = []
        advantage = 0

        for i in reversed(range(len(rewards))):
            if i == len(rewards) - 1:
                next_value = 0 if dones[i] else next_value
            else:
                next_value = values[i + 1]

            delta = rewards[i] + self.gamma * next_value * (1 - dones[i]) - values[i]
            advantage = delta + self.gamma * 0.95 * (1 - dones[i]) * advantage
            advantages.insert(0, advantage)

        return advantages

    def ppo_update(self, query_batch, response_batch, old_log_probs, rewards, values, advantages):
        """Update the policy and value networks using PPO"""
        # Prepare inputs
        inputs_ids = self.tokenizer(query_batch, return_tensors="pt", padding=True).to(self.device)
        response_ids = self.tokenizer(response_batch, return_tensors="pt", padding=True).to(self.device)

        # Forward pass
        outputs = self.policy(**inputs_ids, labels=response_ids.input_ids)
        logits = outputs.logits

        # New log probabilities
        log_probs = -outputs.loss  # Negative of cross-entropy loss

        # Value prediction
        value_outputs = self.policy(**inputs_ids, output_hidden_states=True)
        hidden_states = value_outputs.hidden_states[-1][:, -1, :]  # Last token's hidden state
        value_preds = self.value_head(hidden_states).squeeze(-1)

        # Calculate policy loss with clipping
        ratio = torch.exp(log_probs - old_log_probs)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value loss
        value_targets = rewards
        value_loss = 0.5 * ((value_preds - value_targets) ** 2).mean()

        # Entropy loss to encourage exploration
        entropy_loss = -self.entropy_coef * log_probs.mean()

        # Total loss
        loss = policy_loss + self.value_coef * value_loss + entropy_loss

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(self.policy.parameters()) + list(self.value_head.parameters()),
                                      self.max_grad_norm)
        self.optimizer.step()

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy_loss": entropy_loss.item(),
            "total_loss": loss.item(),
            "approx_kl": (old_log_probs - log_probs).mean().item()
        }

# Step 4: Training Process

class RLTrainer:
    def __init__(self,
                 model_path,
                 tokenizer_path,
                 embedding_model_path="sentence-transformers/all-mpnet-base-v2",
                 lr=5e-5,
                 batch_size=8,
                 epochs=5,
                 wandb_project="rl-after-rag",
                 checkpoint_dir="./rl_model_checkpoints"):

        self.env = RAGEnvironment(model_path, tokenizer_path, embedding_model_path)
        self.reward_model = RewardModel(embedding_model_path)
        self.ppo_trainer = PPOTrainer(model_path, tokenizer_path, lr=lr)

        self.batch_size = batch_size
        self.epochs = epochs
        self.checkpoint_dir = checkpoint_dir

        # Create checkpoint directory if it doesn't exist
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        # Initialize wandb for tracking
        wandb.init(project=wandb_project)

    def prepare_data(self, data_path):
        """Prepare dataset for training"""
        df = pd.read_csv(data_path)
        dataset = Dataset.from_pandas(df)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        return dataloader

    def train(self, data_loader):
        """Run the full RL training loop"""
        best_reward = -float('inf')

        for epoch in range(self.epochs):
            epoch_rewards = []
            epoch_losses = []

            for batch in tqdm(data_loader, desc=f"Epoch {epoch+1}/{self.epochs}"):
                queries = batch['query']
                retrieved_docs_batch = batch['retrieved_docs']
                ground_truths = batch.get('ground_truth', [None] * len(queries))

                # Generate responses with current policy
                responses = []
                for query, docs in zip(queries, retrieved_docs_batch):
                    response = self.env.generate_response(query, docs)
                    responses.append(response)

                # Calculate rewards
                rewards = []
                reward_components = []
                for response, query, docs, gt in zip(responses, queries, retrieved_docs_batch, ground_truths):
                    reward, components = self.reward_model.calculate_reward(response, query, docs, gt)
                    rewards.append(reward)
                    reward_components.append(components)

                # Collect old log probabilities for PPO
                old_log_probs = []
                values = []
                with torch.no_grad():
                    for query, response in zip(queries, responses):
                        inputs = self.ppo_trainer.tokenizer(query, return_tensors="pt").to(self.ppo_trainer.device)
                        response_ids = self.ppo_trainer.tokenizer(response, return_tensors="pt").to(self.ppo_trainer.device)

                        # Get log probabilities
                        outputs = self.ppo_trainer.policy(**inputs, labels=response_ids.input_ids)
                        old_log_prob = -outputs.loss
                        old_log_probs.append(old_log_prob.item())

                        # Get value predictions
                        value_outputs = self.ppo_trainer.policy(**inputs, output_hidden_states=True)
                        hidden_states = value_outputs.hidden_states[-1][:, -1, :]
                        value = self.ppo_trainer.value_head(hidden_states).squeeze(-1)
                        values.append(value.item())

                # Compute advantages
                dones = [True] * len(rewards)  # All episodes end after one step in this setup
                advantages = self.ppo_trainer.compute_advantages(rewards, values, dones)

                # Update policy with PPO
                metrics = self.ppo_trainer.ppo_update(
                    queries, responses,
                    torch.tensor(old_log_probs).to(self.ppo_trainer.device),
                    torch.tensor(rewards).to(self.ppo_trainer.device),
                    torch.tensor(values).to(self.ppo_trainer.device),
                    torch.tensor(advantages).to(self.ppo_trainer.device)
                )

                epoch_rewards.extend(rewards)
                epoch_losses.append(metrics["total_loss"])

                # Log to wandb
                wandb.log({
                    "reward_mean": np.mean(rewards),
                    "reward_std": np.std(rewards),
                    "factual_accuracy": np.mean([comp["factual_accuracy"] for comp in reward_components]),
                    "relevance": np.mean([comp["relevance"] for comp in reward_components]),
                    "conciseness": np.mean([comp["conciseness"] for comp in reward_components]),
                    "policy_loss": metrics["policy_loss"],
                    "value_loss": metrics["value_loss"],
                    "entropy_loss": metrics["entropy_loss"],
                    "total_loss": metrics["total_loss"],
                    "approx_kl": metrics["approx_kl"]
                })

            # End of epoch
            avg_reward = np.mean(epoch_rewards)
            print(f"Epoch {epoch+1}/{self.epochs}: Average Reward = {avg_reward:.4f}, Average Loss = {np.mean(epoch_losses):.4f}")

            # Save checkpoint if better
            if avg_reward > best_reward:
                best_reward = avg_reward
                self.save_checkpoint(f"{self.checkpoint_dir}/best_model")

            # Save periodic checkpoint
            self.save_checkpoint(f"{self.checkpoint_dir}/epoch_{epoch+1}")

    def save_checkpoint(self, path):
        """Save model checkpoint"""
        self.ppo_trainer.policy.save_pretrained(f"{path}_policy")
        self.ppo_trainer.tokenizer.save_pretrained(f"{path}_tokenizer")
        torch.save(self.ppo_trainer.value_head.state_dict(), f"{path}_value_head.pt")
        print(f"Checkpoint saved at {path}")

    def load_checkpoint(self, path):
        """Load model checkpoint"""
        self.ppo_trainer.policy = AutoModelForCausalLM.from_pretrained(f"{path}_policy")
        self.ppo_trainer.tokenizer = AutoTokenizer.from_pretrained(f"{path}_tokenizer")
        self.ppo_trainer.value_head.load_state_dict(torch.load(f"{path}_value_head.pt"))
        self.ppo_trainer.policy.to(self.ppo_trainer.device)
        self.ppo_trainer.value_head.to(self.ppo_trainer.device)
        print(f"Checkpoint loaded from {path}")

# Step 5: Evaluation & Integration

class ModelEvaluator:
    def __init__(self, embedding_model_path="sentence-transformers/all-mpnet-base-v2"):
        self.embedding_model = SentenceTransformer(embedding_model_path)
        self.rouge = evaluate.load('rouge')
        self.bleu = evaluate.load('bleu')

    def evaluate_model(self, model, tokenizer, test_data):
        """Evaluate model performance on test data"""
        results = {
            "rouge1": [],
            "rouge2": [],
            "rougeL": [],
            "bleu": [],
            "relevance": [],
            "factual_accuracy": [],
            "overall_score": []
        }

        reward_model = RewardModel()

        for item in tqdm(test_data, desc="Evaluating"):
            query = item['query']
            retrieved_docs = item['retrieved_docs']
            ground_truth = item.get('ground_truth')

            # Generate response
            inputs = tokenizer(query, return_tensors="pt").to(model.device)
            with torch.no_grad():
                output_ids = model.generate(
                    **inputs,
                    max_length=512,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id else tokenizer.eos_token_id
                )

            response = tokenizer.decode(output_ids[0], skip_special_tokens=True)

            # Calculate metrics
            if ground_truth:
                # ROUGE scores
                rouge_scores = self.rouge.compute(predictions=[response], references=[ground_truth])
                results["rouge1"].append(rouge_scores["rouge1"])
                results["rouge2"].append(rouge_scores["rouge2"])
                results["rougeL"].append(rouge_scores["rougeL"])

                # BLEU score
                bleu_score = self.bleu.compute(predictions=[response.split()], references=[[ground_truth.split()]])
                results["bleu"].append(bleu_score["bleu"])

            # Calculate reward components
            reward, components = reward_model.calculate_reward(response, query, retrieved_docs, ground_truth)
            results["relevance"].append(components["relevance"])
            results["factual_accuracy"].append(components["factual_accuracy"])
            results["overall_score"].append(reward)

        # Calculate averages
        evaluation_results = {metric: np.mean(scores) for metric, scores in results.items() if scores}

        return evaluation_results

    def compare_models(self, base_model, rag_model, rl_model, transfer_model=None, test_data=None):
        """Compare different model versions"""
        models = {
            "Base Model": base_model,
            "RAG Model": rag_model,
            "RL-finetuned Model": rl_model
        }

        if transfer_model:
            models["Transfer-learned Model"] = transfer_model

        comparison = {}
        for name, (model, tokenizer) in models.items():
            print(f"Evaluating {name}...")
            results = self.evaluate_model(model, tokenizer, test_data)
            comparison[name] = results

        # Create comparison report
        report = pd.DataFrame(comparison)
        return report

# Main function to use all components

def main():
    # Example usage
    model_path = "meta-llama/Llama-2-7b-hf"  # Example model
    tokenizer_path = "meta-llama/Llama-2-7b-hf"  # Example tokenizer
    data_path = "WEF_Global_Cooperation_Barometer_2025.pdf"  # Path to your data

    # Initialize trainer
    trainer = RLTrainer(
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        lr=5e-5,
        batch_size=8,
        epochs=5,
        wandb_project="rl-after-rag",
        checkpoint_dir="./rl_model_checkpoints"
    )

    # Prepare data
    data_loader = trainer.prepare_data(data_path)

    # Train model
    trainer.train(data_loader)

    # Evaluate final model
    evaluator = ModelEvaluator()

    # Load models for comparison
    base_model = AutoModelForCausalLM.from_pretrained("base_model_path")
    base_tokenizer = AutoTokenizer.from_pretrained("base_tokenizer_path")

    rag_model = AutoModelForCausalLM.from_pretrained("rag_model_path")
    rag_tokenizer = AutoTokenizer.from_pretrained("rag_tokenizer_path")

    rl_model = AutoModelForCausalLM.from_pretrained("./rl_model_checkpoints/best_model_policy")
    rl_tokenizer = AutoTokenizer.from_pretrained("./rl_model_checkpoints/best_model_tokenizer")

    # Example test data
    test_data = [
        {
            "query": "What are the benefits of reinforcement learning in LLMs?",
            "retrieved_docs": ["Document about RL benefits in LLMs", "Paper on RL applications"],
            "ground_truth": "Reinforcement learning in LLMs improves alignment, reduces hallucinations, and enhances response quality through feedback-based optimization."
        },
        # Add more test examples
    ]

    # Compare models
    comparison_report = evaluator.compare_models(
        base_model=(base_model, base_tokenizer),
        rag_model=(rag_model, rag_tokenizer),
        rl_model=(rl_model, rl_tokenizer),
        test_data=test_data
    )

    print("Model Comparison Report:")
    print(comparison_report)

    # Save final report
    comparison_report.to_csv("model_comparison_results.csv")

    # Close wandb
    wandb.finish()

if __name__ == "__main__":
    main()

