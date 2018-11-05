def print_transitions(transition):
    print('obs ',len(transition), transition[0].shape)
    print('obs_full ', len(transition), transition[1].shape)#, len(transition[1][0][0]))
    print('actions ', len(transition),transition[2].shape)
    print('rewards ', len(transition), transition[3].shape)#, len(transition[1][0][0]))
    print('next obs ', len(transition), transition[4].shape)
    print('next obs full ', len(transition), transition[5].shape)#, len(transition[1][0][0]))
    print('dones ', len(transition), transition[6].shape)#, len(transition[1][0][0]))
    #print(transition[6])
def view_replay_buffer_data(replay_buffer):
    if len(replay_buffer) > 0:
        transition = replay_buffer.memory[0]
    print_transitions(transition)