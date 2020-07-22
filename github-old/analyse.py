

if __name__=="__main__":
    sequences, sequenceStats = get_sequence_stats()
    write_sequence_data_to_file(sequences, sequenceStats)