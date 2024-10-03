from typing import List, Tuple
import csv
from model import MusicHighlighter
from lib import *
import tensorflow.compat.v1 as tf
import numpy as np
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ''

def convert_to_seconds(time_str):
    minutes, seconds = time_str.split(':')
    total_seconds = int(minutes) * 60 + int(seconds)
    return total_seconds

def overlap(interval1: Tuple[int, int], interval2: Tuple[int, int]) -> int:
    start = max(interval1[0], interval2[0])
    end = min(interval1[1], interval2[1])
    return max(0, end - start)

def calculate_metrics(predictions: List[Tuple[int, int]], true_labels: List[Tuple[int, int]]) -> Tuple[float, float, float]:
    print(predictions)
    print(true_labels)
    indexPrediction = 0
    indexLabel = 0
    maxOverlap = 0

    for i in range(len(predictions)):
        for j in range(len(true_labels)):
            if overlap(predictions[i], true_labels[j]) > maxOverlap:
                maxOverlap = overlap(predictions[i], true_labels[j])
                indexPrediction = i
                indexLabel = j
       
    precision = maxOverlap / (predictions[indexPrediction][1] - predictions[indexPrediction][0])
    recall = maxOverlap / (true_labels[indexLabel][1] - true_labels[indexLabel][0]) 
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision, recall, f1_score


def evaluate(file_path, length=15):
    PRE = 0
    REC = 0
    F1 = 0

    with tf.Session() as sess:
        model = MusicHighlighter()
        sess.run(tf.global_variables_initializer())
        model.saver.restore(sess, 'model/model')

        count = 0

        with open(file_path, mode='r', encoding='latin-1') as file:
            reader = csv.reader(file)
            next(reader) 
            for row in reader:
                if (row[1] != "La 1 thang con trai"):
                    continue
                f = "dataset\\" + row[1] + ".mp3"
                # print(f)
                audio, spectrogram, duration = audio_read(f)
                n_chunk, remainder = np.divmod(duration, 3)
                chunk_spec = chunk(spectrogram, n_chunk)
                pos = positional_encoding(batch_size=1, n_pos=n_chunk, d_pos=model.dim_feature*4)
                
                n_chunk = n_chunk.astype('int')
                chunk_spec = chunk_spec.astype('float')
                pos = pos.astype('float')
                
                attn_score = model.calculate(sess=sess, x=chunk_spec, pos_enc=pos, num_chunk=n_chunk)
                attn_score = np.repeat(attn_score, 3)
                attn_score = np.append(attn_score, np.zeros(remainder))

                attn_score = attn_score / attn_score.max()
                

                attn_score = attn_score.cumsum()
                attn_score = np.append(attn_score[length], attn_score[length:] - attn_score[:-length])
                # index = np.argmax(attn_score)

                # predictions = [index, index+length]
                
                sorted_indices = np.argsort(attn_score)[::-1]
                predictions = []
                used_indices = set()

                for index in sorted_indices:
                    if (len(predictions) == 3):
                        break
                    is_overlap = any(index < h_end and index + length > h_start for (h_start, h_end) in predictions)

                    if not is_overlap:
                        predictions.append([index, index + length])

                true_labels: List[Tuple[int, int]] = []
                start1 = convert_to_seconds(row[4])
                start2 = convert_to_seconds(row[5])
                start3 = convert_to_seconds(row[6])
                true_labels.append((start1, start1 + 15)) 
                true_labels.append((start2, start2 + 15))  
                true_labels.append((start3, start3 + 15))

                
                count += 1
                print(count)
                precision, recall, f1_score = calculate_metrics(predictions, true_labels)
                PRE += precision
                REC += recall
                F1 += f1_score
                print(f"Precision: {precision:.2f}")
                print(f"Recall: {recall:.2f}")
                print(f"F1 Score: {f1_score:.2f}")    

        PRE /= count
        REC /= count
        F1 /= count

        print(f"Precision: {PRE:.2f}")
        print(f"Recall: {REC:.2f}")
        print(f"F1 Score: {F1:.2f}")        
                    
                    
if __name__ == '__main__':
    file_path = 'dataset\\label.csv'
    evaluate(file_path=file_path, length=30)