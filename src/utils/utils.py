import io
import chess
import torch
from .bot import *
from transformers import AutoModel


# Convert FEN string to a chess.Board object
def fen_to_board(fen: str) -> chess.Board:
    board = chess.Board(fen)
    return board


# Convert PGN text to a chess.Board object
def pgn_to_board(pgn: str) -> chess.Board:
    game = chess.pgn.read_game(io.StringIO(pgn))
    return game.end().board()


# Load model from Hugging Face
def load_model_from_hf(repo_id):
    hf_model = AutoModel.from_pretrained(
        repo_id,
        cache_dir="./.model_cache"
    )
    model = EvalNet()
    model.load_state_dict(hf_model.state_dict())
    model.eval()
    return model


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
        try:
            score = minimax(board, depth - 1, False, model)
        except Exception as e:
            print("Error during minimax evaluation:", e)
            score = float('-inf')
        board.pop()
        if score > best_score:
            best_score = score
            best_move = move
    return best_move
