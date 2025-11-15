# Necessary imports
from .utils import chess_manager, GameContext, EvalNet, encode_board
import torch
import os
from .utils import pgn_to_board


# Write code here that runs once
# Can do things like load models from huggingface, make connections to subprocesses, etcwenis


# Minimax search using trained model
def minimax(board, depth, maximizing, model):
    # Base case: evaluate board using neural network
    if depth == 0 or board.is_game_over():
        x = encode_board(board)
        with torch.no_grad():
            return model(x).item()
    moves = list(board.legal_moves)
    # Recursive case: explore moves
    if maximizing:
        max_eval = float('-inf')
        for move in moves:
            board.push(move)
            eval_score = minimax(board, depth - 1, False, model)
            board.pop()
            if eval_score > max_eval:
                max_eval = eval_score
        return max_eval
    # Minimizing player
    else:
        min_eval = float('inf')
        for move in moves:
            board.push(move)
            eval_score = minimax(board, depth - 1, True, model)
            board.pop()
            if eval_score < min_eval:
                min_eval = eval_score
        return min_eval


# Select the best move using minimax and the neural network
def make_best_move(board, model, depth):
    best_move = None
    best_score = float('-inf')
    for move in board.legal_moves:
        board.push(move)
        score = minimax(board, depth - 1, False, model)
        board.pop()
        if score > best_score:
            best_score = score
            best_move = move
    return best_move


# Load model weights
def load_model(path="eval_net_weights.pt"):
    model = EvalNet()
    if os.path.exists(path):
        model.load_state_dict(torch.load(path))
        print(f"Loaded model weights from {path}")
    else:
        print("Model weights not found, initializing fresh model")
    model.eval()
    return model


# Entrypoint function called to get the next move
@chess_manager.entrypoint
def get_move(pgn: str) -> str:
    board = pgn_to_board(pgn)

    # Get or create cached model
    ctx = chess_manager.context
    model = ctx.state.get('model')
    if model is None:
        model = load_model()
        ctx.state['model'] = model

    best_move = make_best_move(board, model, depth=3)
    return best_move.uci() if best_move else None


# Reset function called at the start of each new game
@chess_manager.reset
def reset_func(ctx: GameContext):
    model = load_model()
    ctx.state['model'] = model
