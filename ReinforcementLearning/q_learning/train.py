import h5py
from dlgo.agent import q
from dlgo.rl import experience


board_size = (5,5)
learning_agent_filename = '/home/bart/Q_output_file1.h5'
exp_filename = '/home/bart/Q_experience_file.h5'

learning_agent = q.load_q_agent(h5py.File(learning_agent_filename), board_size)

exp_buffer = experience.load_experience(h5py.File(exp_filename))

learning_agent.train(exp_buffer, batch_size=8)

with h5py.File('/home/bart/Q_updated_agent.h5', 'w') as updated_agent_outf:
    learning_agent.serialize(updated_agent_outf)

