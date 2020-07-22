import random
import statistics
import csv
from collections import OrderedDict, defaultdict
import math

path_to_csv = '/content/gdrive/My Drive/VISUM2020/visum-competition2020-master/visum-code/dataset/train/labels.csv'

def seq_indices_to_frame_indices(seq_indices, first_frame = 0, last_frame = 59):
    # Get all indices for one sequence list
    index_list = []
    for seq_id in seq_indices:
        index_list += list(range(60*seq_id + first_frame, 60*seq_id + last_frame + 1))
    return index_list


def get_sequence_stats():
    # Get all the annotations
    sequences = OrderedDict()
    with open(path_to_csv, newline='') as csvfile:
        reader = csv.reader(csvfile, delimiter=';')
        for i, row in enumerate(reader):
            if i==0: # header
                continue
            if len(row) == 3:
                boxes = eval(row[2])
            else:
                boxes = []
            sequences.setdefault(int(row[0]), []).append([int(row[1]), boxes])

    # Get data for each sequence (in order):
    # No. of frames with fish
    # Avg number of frames with fish
    # Max number of fish in frame
    # Min number of fish in frame
    # Number of frames with each frequency of fishes
    # Avg fish size
    # Max fish size
    # Min fish size
    sequenceStats = []
    for seqNo, frames in sequences.items():
        frames.sort(key=lambda x: x[0])
        nFramesWithFishes = 0
        nFishes = 0
        minNFish = math.inf
        maxNFish = 0
        fishCountFreq = defaultdict(int)

        totalFishArea = 0
        minAreaFish = math.inf
        minAreaFishDims = []
        maxAreaFish = 0
        maxAreaFishDims = []
        for frame in frames:
            boxes = frame[1]
            if len(boxes) > 0:
                nFramesWithFishes += 1

            nFishes += len(boxes)
            if len(boxes) < minNFish:
                minNFish = len(boxes)
            if len(boxes) > maxNFish:
                maxNFish = len(boxes)
            fishCountFreq[len(boxes)] += 1

            for x_min, y_min, x_max, y_max in boxes:
                dx = x_max - x_min
                dy = y_max - y_min
                area = dx*dy
                if area < minAreaFish:
                    minAreaFish = area
                    minAreaFishDims = [dx,dy]
                if area > maxAreaFish:
                    maxAreaFish = area
                    maxAreaFishDims = [dx,dy]
                totalFishArea += area

        fishCountFreq = sorted(fishCountFreq.items()) 

        # Print the values for the sequences
        #print("Sequence %s: " %(seqNo))
        #print("     %s frames with fishes." %(nFramesWithFishes))
        #print("     Average number of fish per frame: %s." %(nFishes/len(frames)))
        #print("     Minimum number of fish in a frame: %s." %(minNFish))
        #print("     Maximum number of fish in a frame: %s." %(maxNFish))
        #print("     Fish count frequencies: %s." %(fishCountFreq))
        #print("     Average fish area: %s." %(totalFishArea/nFishes))
        #print("     Smallest fish area: %s (%s x %s)." %(minAreaFish, minAreaFishDims[0], minAreaFishDims[1]))
        #print("     Largest fish area: %s (%s x %s)." %(maxAreaFish, maxAreaFishDims[0], maxAreaFishDims[1]))

        # Store all the sequence data
        sequenceStats.append([seqNo, nFramesWithFishes, nFishes/len(frames), minNFish, maxNFish, fishCountFreq, totalFishArea/nFishes, minAreaFish, minAreaFishDims[0], minAreaFishDims[1], maxAreaFish, maxAreaFishDims[0], maxAreaFishDims[1]])
        
    return sequences, sequenceStats

def write_sequence_data_to_file(sequences, sequenceData):
    # Write the sequence results to a file
    with open('sequence_statistics.csv', 'w', newline='') as file:
        writer = csv.writer(file, delimiter=";")
        writer.writerow(['Seq #', 'No. frames w/fish', 'Avg. fish per frame', 'Min. fish', 'Max. fish', 'Fish count frequencies', 'Avg. fish area', 'Min. fish area', 'Min. fish width', 'Min. fish height', 'Max. fish area', 'Max. fish width', 'Max. fish height'])
        writer.writerows(sequenceData)

    # Write the fish frequency matrix to a file
    with open('fish_frequency_matrix.csv', 'w', newline='') as file:
        writer = csv.writer(file, delimiter=";")
        writer.writerow(['Seq # / Frame #'] + list(range(1, 60)))
        for seqNo, frames in sequences:
            row = [seqNo]
            for frame in frames:
                row.append(len(frame[1]))
            writer.writerow(row)

def get_set_distributions(sequence_subset):
    fishNo = [x[2] for x in sequence_subset]
    fishSize = [x[6] for x in sequence_subset]
    return [statistics.mean(fishNo), statistics.stdev(fishNo), statistics.mean(fishSize), statistics.stdev(fishSize)]

def is_close_to_overall_distribution(sequence_distributions, subset_distributions, distribution_tolerances):
    for sequence_stat, subset_stat, stat_tol in zip(sequence_distributions,subset_distributions,distribution_tolerances):
            if(abs(sequence_stat-subset_stat) > stat_tol):
                return False
    return True

def split_data(sequenceStats):
    random.seed(4)
    well_shuffled = False
    while(not(well_shuffled)):
        random.shuffle(sequenceStats)
        training_set = sequenceStats[:41] # 41 = 84*0.7*0.7
        validation_set = sequenceStats[41:59] # 18 = 84*0.7*0.3
        testing_set = sequenceStats[59:] # 25
        
        sequence_distributions = get_set_distributions(sequenceStats)
        training_distributions = get_set_distributions(training_set)
        validation_distributions = get_set_distributions(validation_set)
        testing_distributions = get_set_distributions(testing_set)
        all_subset_distributions = [training_distributions, validation_distributions, testing_distributions]

        training_indices = [x[0] for x in training_set]
        validation_indices = [x[0] for x in validation_set]
        testing_indices = [x[0] for x in testing_set]

        distribution_tolerances = [0.1,0.15,1000,1500]
        well_shuffled = True
        for subset_distributions in all_subset_distributions:
            if(not(is_close_to_overall_distribution(sequence_distributions, subset_distributions, distribution_tolerances))):
                well_shuffled = False
    
    #print("Final distributions:")
    #print(sequence_distributions)
    #print(training_distributions)
    #print(validation_distributions)
    #print(testing_distributions)

    return training_indices, validation_indices, testing_indices

if __name__=="__main__":
    sequences, sequenceStats = get_sequence_stats()
    training_indices, validation_indices, testing_indices = split_data(sequenceStats)

    sequences = list(sequences.items())
    shuffledSequences = [sequences[idx] for idx in training_indices] + [sequences[idx] for idx in validation_indices] + [sequences[idx] for idx in testing_indices]
    print(training_indices)
    print(validation_indices)
    print(testing_indices)
    write_sequence_data_to_file(shuffledSequences, sequenceStats)

