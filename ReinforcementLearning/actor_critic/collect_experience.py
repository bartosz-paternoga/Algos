import h5py
import time
from dlgo.encoders import oneplane
from dlgo.networks import large
from dlgo.agent import ac
from dlgo.agent import naive
from dlgo import goboard_slow as goboard
from dlgo import gotypes
from dlgo import scoring
from dlgo.rl import experience

from keras.models import Model
from keras.layers import Conv2D, Dense, Flatten, Input



board_size = (5,5)
encoder = oneplane.OnePlaneEncoder(board_size)

board_input = Input(shape=encoder.shape(), name='board_input')

processed_board = board_input   
for layer in large.layers(encoder.shape()):
    processed_board = layer(processed_board)

policy_hidden_layer = Dense(512, activation='relu')(processed_board)
policy_output = Dense(encoder.num_points(), activation='softmax')(policy_hidden_layer)

value_hidden_layer = Dense(512, activation='relu')(processed_board)
value_output = Dense(1, activation='tanh')(value_hidden_layer)

model = Model(inputs=[board_input], outputs=[policy_output, value_output])




agent1 = ac.ACagent(model, encoder)
agent2 = ac.ACagent(model, encoder)

collector1 = experience.ExperienceCollector()
collector2 = experience.ExperienceCollector()
agent1.set_collector(collector1)
agent2.set_collector(collector2)

with h5py.File('/home/bart/AC_output_file1.h5', 'w') as outf:
    agent1.serialize(outf)
with h5py.File('/home/bart/AC_output_file2.h5', 'w') as outf:
    agent2.serialize(outf)


a = []

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
    board_size = 5
    game = goboard.GameState.new_game(board_size)
    bots = {
        #gotypes.Player.black: naive.RandomBot(),
        gotypes.Player.black: agent1,
        gotypes.Player.white: agent2,
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
    a.append(game_result.winner)
    return game_result.winner


num_games = 10

for i in range(num_games):
    collector1.begin_episode()
    collector2.begin_episode()
    
    game_record = main()

    if a[0] == 'Player.black':
        collector1.complete_episode(reward=1)
        collector2.complete_episode(reward=-1)
    else:
        collector2.complete_episode(reward=1)
        collector1.complete_episode(reward=-1)

exp = experience.combine_experience([collector1,collector2])

with h5py.File('/home/bart/AC_experience_file.h5', 'w') as experience_outf:
    exp.serialize(experience_outf)

