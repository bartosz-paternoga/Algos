import h5py
import time
from collections import namedtuple
from dlgo.gotypes import Player
from dlgo.agent import ac
from dlgo.agent import naive
from dlgo import goboard_slow as goboard
from dlgo import gotypes
from dlgo import scoring



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

class GameRecord(namedtuple('GameRecord', 'winner margin')):
    pass

def simulate_game(black_player, white_player):
    board_size = 5
    game = goboard.GameState.new_game(board_size)
    bots = {
        #gotypes.Player.black: naive.RandomBot(),
        gotypes.Player.black: black_player,
        gotypes.Player.white: white_player,
        #gotypes.Player.white: agent3,
        
    }
    while not game.is_over():
        time.sleep(0.3)  # <1>

        print(chr(27) + "[2J")  # <2>
        print_board(game.board)
        bot_move = bots[game.next_player].select_move(game)
        print_move(game.next_player, bot_move)
        game = game.apply_move(bot_move)
    
    game_result = scoring.compute_game_result(game)
    print("and the winner is:",game_result.winner)

    return GameRecord(
        #moves=moves,
        winner=game_result.winner,
        margin=game_result.winning_margin,
    )



board_size = (5,5)

agent1 = ac.load_ac_agent(h5py.File('/home/bart/AC_updated_agent.h5'), board_size)
agent2 = ac.load_ac_agent(h5py.File('/home/bart/AC_output_file1.h5'), board_size)

collector1 = None
collector2 = None
agent1.set_collector(collector1)
agent2.set_collector(collector2)

num_games = 10
wins = 0
losses = 0
color1 = Player.black

for i in range(num_games):
    print('Simulating game %d/%d...' % (i + 1, num_games))
    if color1 == Player.black:
        black_player, white_player = agent1, agent2
    else:
        white_player, black_player = agent1, agent2
    game_record = simulate_game(black_player, white_player)
    if game_record.winner == color1:
        wins += 1
    else:
        losses += 1
    color1 = color1.other
print('Agent 1 record: %d/%d' % (wins, wins + losses)) 



