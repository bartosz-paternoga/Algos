import h5py

from dlgo import agent
from dlgo.agent import naive
from dlgo import goboard_slow as goboard
from dlgo import gotypes
import time

from dlgo.agent.predict import DeepLearningAgent, load_prediction_agent

go_board_rows, go_board_cols = 19, 19

model_fileX = h5py.File("/home/bart/dlgo/small_model_epoch_5.h5", "r")

bot_from_file = load_prediction_agent(model_fileX,go_board_rows, go_board_cols)


COLS = 'ABCDEFGHJKLMNOPQRST'
STONE_TO_CHAR = {
None: ' . ',
gotypes.Player.black: ' x ',
gotypes.Player.white: ' o ',
}


def print_move(player, move):
    if move.is_pass:
        move_str = 'passes'
    elif move.is_resign:
        move_str = 'resigns'
    else:
        move_str = '%s%d' % (COLS[move.point.col - 1], move.point.row)
    print('%s %s' % (player, move_str))
    
def print_board(board):
    for row in range(board.num_rows, 0, -1):
        bump = " " if row <= 9 else ""
        line = []
        for col in range(1, board.num_cols + 1):
            stone = board.get(gotypes.Point(row=row, col=col))
            line.append(STONE_TO_CHAR[stone])
        print('%s%d %s' % (bump, row, ''.join(line)))
    print(' ' + ' '.join(COLS[:board.num_cols]))

def main():
    board_size = 19
    game = goboard.GameState.new_game(board_size)
    bots = {
        gotypes.Player.black: agent.naive.RandomBot(),
        #gotypes.Player.white: agent.naive.RandomBot(),
        #gotypes.Player.black: bot_from_file,
        gotypes.Player.white: bot_from_file,
        
    }
    while not game.is_over():
        time.sleep(0.3)  # <1>

        print(chr(27) + "[2J")  # <2>
        print_board(game.board)
        bot_move = bots[game.next_player].select_move(game)
        print_move(game.next_player, bot_move)
        game = game.apply_move(bot_move)
    

    print("and the winner is:",game.winner())


main()

# <1> We set a sleep timer to 0.3 seconds so that bot moves aren't printed too fast to observe
# <2> Before each move we c
 
