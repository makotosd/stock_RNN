#!/usr/bin/python
# coding: utf-8

from time import sleep
import queue_task
import train


def main():
    BATCH_SIZE = 32
    dequeued_task = queue_task.Task()
    if not dequeued_task.is_empty:
        print(dequeued_task.cc, dequeued_task.target_feature, dequeued_task.rnn,
              dequeued_task.num_of_neuron, BATCH_SIZE, dequeued_task.num_train)
        # sleep(10)
        train.train(dequeued_task.cc.split(','), dequeued_task.target_feature, dequeued_task.rnn,
                    dequeued_task.num_of_neuron, BATCH_SIZE, dequeued_task.num_train)
        dequeued_task.update_to_finished()
    del dequeued_task

if __name__ == "__main__" :
    main()