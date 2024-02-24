import os
import csv
import tensorflow as tf

dir_source_path = 'logs/ex2/'
dir_target_path = 'logs/ex2_tf/'
csvs = os.listdir(dir_source_path)

for csv_file in csvs:
    with open(dir_source_path + csv_file) as file_object:
        train_summary_writer = tf.summary.create_file_writer(dir_target_path + csv_file)
        train_summary_writer.set_as_default()

        reader = csv.reader(file_object)
        for i, line in enumerate(reader):
            if i != 0:
                timestamp, step, reward, avg_reward, loss = line
                tf.summary.scalar("Average Reward (Last 100 Episodes)", float(avg_reward), int(step))
                tf.summary.scalar("Reward Per Episode", float(reward), int(step))
                tf.summary.scalar("Loss Per Episode", float(loss), int(step))
