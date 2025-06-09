# training/train_atis.py
from models.atis.reward_model import RewardModel
from models.atis.opponent_model import OpponentModel
from models.atis.planner import StrategicPlanner
from utils.dataloader import load_training_data
from training.logger import Logger

import yaml

def train_atis(config_path="training/config.yaml"):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    reward_model = RewardModel(config["reward"])
    opponent_model = OpponentModel()
    planner = StrategicPlanner(config["planner"])
    logger = Logger()

    train_loader = load_training_data(config["data"])

    for epoch in range(config["training"]["epochs"]):
        for batch in train_loader:
            opponent_pred = opponent_model.predict_action(batch["opponent_state"])
            plan = planner.generate_plan(batch["state"], opponent_pred)
            reward = reward_model.compute_reward(batch["state"], plan)

        logger.log(f"Epoch {epoch+1} - Sample Reward: {reward}")

if __name__ == "__main__":
    train_atis()
