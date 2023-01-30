from time import time


def start_timer(agent):
    agent.timer[_target(agent)] = time()


def end_timer(agent):
    target = _target(agent)
    setattr(agent,
            f'{target}_duration',
            time() - agent.timer[target])


def _target(agent):
    for event in agent.timer.keys():
        if event in agent.event:
            return event
