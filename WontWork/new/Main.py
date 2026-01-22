# --- main.py (דוגמה) ---
import torch
# הגדרת היפר-פרמטרים
config = {
    'device': 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu',
    'learning_rate': 0.001,
    'weight_decay': 0.0001,
    'buffer_size': 50000,
    'batch_size': 128,
    'temperature': 1.0,
    'self_play_games_per_iter': 50,
    'training_steps_per_iter': 100,
    'save_interval': 10,
}

# יצירת כל הרכיבים
# action_space = ... (צריך לבנות מחלקה שתנהל את מרחב הפעולות)
# env = PegSolitaireEnv(...)
# model = PegSolitaireNetV2(...)

# יצירת המאמן והרצת הלולאה
# trainer = AlphaZeroTrainer(model, env, mcts_simulations=100, action_space=action_space, config=config)
# trainer.train_loop(num_iterations=1000)