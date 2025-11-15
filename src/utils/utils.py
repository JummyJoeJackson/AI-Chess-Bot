import chess
import io

# Convert PGN text to a chess.Board object
def pgn_to_board(pgn: str) -> chess.Board:
    game = chess.pgn.read_game(io.StringIO(pgn))
    return game.end().board()
