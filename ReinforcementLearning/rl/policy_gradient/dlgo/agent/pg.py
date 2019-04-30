"""Policy gradient learning."""
import numpy as np
from keras import backend as K
from keras.optimizers import SGD

from dlgo.agent.helpers import is_point_an_eye
from dlgo.encoders import oneplane
#from dlgo import encoders
from dlgo import goboard_slow as goboard
from dlgo import kerasutil


class PolicyAgent():
    """An agent that uses a deep policy network to select moves."""
    def __init__(self, model, encoder):
        self._model = model
        self._encoder = encoder

    def serialize(self, h5file):
        h5file.create_group('encoder')
        h5file['encoder'].attrs['name'] = self._encoder.name()
        h5file['encoder'].attrs['board_width'] = self._encoder.board_width
        h5file['encoder'].attrs['board_height'] = self._encoder.board_height
        h5file.create_group('model')
        kerasutil.save_model_to_hdf5_group(self._model, h5file['model'])

    def set_collector(self, collector):
        self._collector = collector

    def select_move(self, game_state):

        board_tensor = self._encoder.encode(game_state)
        x = np.array([board_tensor])
        move_probs = self._model.predict(x)[0]

        # Prevent move probs from getting stuck at 0 or 1.
        eps = 1e-5
        move_probs = np.clip(move_probs, eps, 1 - eps)
        # Re-normalize to get another probability distribution.
        move_probs = move_probs / np.sum(move_probs)

        # Turn the probabilities into a ranked list of moves.
        num_moves = self._encoder.board_width * self._encoder.board_height
        candidates = np.arange(num_moves)
        ranked_moves = np.random.choice(candidates, num_moves, replace=False, p=move_probs)
        
        for point_idx in ranked_moves:
            point = self._encoder.decode_point_index(point_idx)
            if game_state.is_valid_move(goboard.Move.play(point)) and \
                    not is_point_an_eye(game_state.board, point, game_state.next_player):
                if self._collector is not None:
                    self._collector.record_decision(state=board_tensor, action=point_idx)
                return goboard.Move.play(point)
        # No legal, non-self-destructive moves less.
        return goboard.Move.pass_turn()

    def train(self, experience, lr=0.0000001, clipnorm=1.0, batch_size=512):
        opt = SGD(lr=lr, clipnorm=clipnorm)
        self._model.compile(loss='categorical_crossentropy', optimizer=opt)

        n = experience.states.shape[0]
        # Translate the actions/rewards.
        num_moves = self._encoder.board_width * self._encoder.board_height
        y = np.zeros((n, num_moves))
        for i in range(n):
            action = experience.actions[i]
            reward = experience.rewards[i]
            y[i][action] = reward

        self._model.fit(
            experience.states, y,
            batch_size=batch_size,
            epochs=1)    


def load_policy_agent(h5file,board_size):
    model = kerasutil.load_model_from_hdf5_group(
        h5file['model'])
        #custom_objects={'policy_gradient_loss': policy_gradient_loss})
    encoder_name = h5file['encoder'].attrs['name']
    if not isinstance(encoder_name, str):
        encoder_name = encoder_name.decode('ascii')
    board_width = h5file['encoder'].attrs['board_width']
    board_height = h5file['encoder'].attrs['board_height']
    encoder = oneplane.OnePlaneEncoder(board_size)
    return PolicyAgent(model, encoder) 

