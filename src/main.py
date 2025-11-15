from .utils import *


# Set repo_id for model loading
repo_id = "JummyJoeJackson/chess-bot-model"


# Entrypoint function called to get the next move
@chess_manager.entrypoint
def get_move(ctx: GameContext) -> str:
    board = ctx.board
    
    """
    # Get or create cached model
    ctx = chess_manager.context
    model = ctx.state.get('model')
    if model is None:
        try:
            model = load_model_from_hf()
        except Exception as e:
            print("Failed to load model from HF Hub, initializing fresh model", e)
            model = EvalNet()
        ctx.state['model'] = model
    
   
    try:
        best_move = make_best_move(board, model, depth=3)
    except Exception as e:
        print("Error during move calculation:", e)
    
    """

    # Fallback: make a random legal move
    best_move = list(board.legal_moves)[0]

    # Ensure best_move is not None
    if not best_move:
        raise ValueError("No legal moves available")
    return best_move


# Reset function called at the start of each new game
@chess_manager.reset
def reset_func(ctx: GameContext):
    try:
        model = load_model_from_hf()
    except Exception as e:
        print("Failed to load model from HF Hub in reset, initializing fresh model", e)
        model = EvalNet()
    ctx.state['model'] = model
