import magenta.music as mm
from magenta.models.music_vae import TrainedModel
from magenta.models.music_vae import configs
import numpy as np
import os
import argparse

tm = TrainedModel(
  config=configs.CONFIG_MAP['cat-mel_2bar_big'],
  batch_size=512,  # From config
  checkpoint_dir_or_path='cat-mel_2bar_big.tar',
)

def encode_midi(input_midi_path, output_path, print_progress=False):
  """
  Convert midi file to latent vector file.
  If the midi file fails to convert to a NoteSequence, or the NoteSequence
    does not correspond to any tensors, do not output any file.
  input_midi_path: Path to midi file.
  output_path: Path to .npy file to save latent vectors.
    The output latent vector array has shape (# NoteSequences, 
    # latent vector dimensions)
  """
  try:
    ns = mm.midi_file_to_note_sequence(input_midi_path)
  except mm.MIDIConversionError as e:
    print(input_midi_path, 'Midi conversion error:', str(e))
    return
  tensors = tm._config.data_converter.to_tensors(ns)
  if len(tensors.inputs) == 0 and print_progress:
    print(input_midi_path, 'does not encode to any vectors')
    return
  _, mu, _ = tm.encode_tensors(
    list(tensors.inputs),
    list(tensors.lengths),
    list(tensors.controls))
  np.save(output_path, mu)
  if print_progress:
    print('Encoded', input_midi_path, '->', len(tensors.inputs), 'vectors at', output_path)


parser = argparse.ArgumentParser(
  description='Encode MIDI files to latent vectors.')
parser.add_argument('--input_dir',
  help='Input directory to recursively look for MIDI files')
parser.add_argument('--output_dir',
  help='Output directory to write latent vector files. The directory structure of input_dir will be replicated')
parser.add_argument('--print_progress', action='store_true', 
  help='Print encoding progress')
args = parser.parse_args()


for dirpath, _, filenames in os.walk(args.input_dir):
  for filename in filenames:
    if filename.endswith('.mid'):
      input_path = os.path.join(dirpath, filename)
      # Create dir with output file
      output_vecs_dir = os.path.join(
          args.output_dir,
          os.path.relpath(dirpath, start=args.input_dir))
      os.makedirs(output_vecs_dir, exist_ok=True)
      output_path = os.path.join(
          output_vecs_dir,
          os.path.splitext(filename)[0] + '.npy')
      encode_midi(input_path, output_path, args.print_progress)