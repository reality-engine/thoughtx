# def segment_eeg_data(eeg_data: np.ndarray, segment_length: int = 128) -> np.ndarray:
#     """
#     Segment the EEG data into chunks of specified length.

#     Parameters:
#     - eeg_data: The preprocessed EEG data.
#     - segment_length: The length of each segment.

#     Returns:
#     - Segmented EEG data.
#     """
#     # Calculate the number of segments based on the provided segment length
#     num_segments = eeg_data.shape[1] // segment_length

#     # List to store segmented chunks
#     segmented_data = []

#     # Loop through and create segments
#     for i in range(num_segments):
#         start_idx = i * segment_length
#         end_idx = (i + 1) * segment_length
#         segment = eeg_data[:, start_idx:end_idx]
#         segmented_data.append(segment)

#     return np.array(segmented_data)


# def prepare_input_sample_for_inference(eeg_data: np.ndarray, tokenizer, max_len: int = 56) -> dict:
#     """
#     Prepare an input sample suitable for the EEG-to-text model.

#     Parameters:
#     - eeg_data: The preprocessed EEG data.
#     - tokenizer: The BART tokenizer.
#     - max_len: The maximum length for tokenization.

#     Returns:
#     - A dictionary containing the prepared sample.
#     """
#     # Tokenize a placeholder text ("<s>") as a seed to generate the subsequent tokens (text)
#     target_ids = tokenizer.encode("<s>", return_tensors="pt", max_length=max_len, pad_to_max_length=True)

#     # Preparing the sample dictionary
#     input_sample = {
#         "target_ids": target_ids,
#         "sent_level_EEG": torch.tensor(eeg_data, dtype=torch.float32)  # Converting the preprocessed EEG data to a torch tensor
#     }

#     return input_sample
