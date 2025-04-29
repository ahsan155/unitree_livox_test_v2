# Copyright 2024 The human_scene_transformer Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Evaluates Model on Pedestrian dataset."""

import os
from typing import Sequence
from absl import app
from absl import flags
from absl import logging
import gin
from human_scene_transformer.model import model as hst_model
from human_scene_transformer.model import model_params
from human_scene_transformer.pedestrians import dataset_params
from human_scene_transformer.pedestrians import input_fn
import numpy as np
import tensorflow as tf
import tqdm


_MODEL_PATH = flags.DEFINE_string(
    'model_path',
    None,
    'Path to model directory.',
)

_CKPT_PATH = flags.DEFINE_string(
    'checkpoint_path',
    None,
    'Path to model checkpoint.',
)


def min_ade(pred, gt):
  dist = np.linalg.norm(pred - gt[..., tf.newaxis, :], axis=-1)
  ade = np.mean(dist, axis=0)
  minade = np.min(ade, axis=-1)
  return minade


def min_fde(pred, gt):
  fde = np.linalg.norm(pred[-1] - gt[-1, ..., tf.newaxis, :], axis=-1)
  minfde = np.min(fde, axis=-1)
  return minfde


def evaluation(checkpoint_path):
  """Evaluates Model on Pedestrian dataset."""
  d_params = dataset_params.PedestriansDatasetParams(
      num_agents=None,
      eval_config='test'
      )

  dataset = input_fn.load_dataset(
      d_params,
      split='test',
      augment=False,
      shuffle=False,
      repeat=False,
      deterministic=True,
  )

  model_p = model_params.ModelParams()

  model = hst_model.HumanTrajectorySceneTransformer(model_p)

  _, _ = model(next(iter(dataset.batch(1))), training=False)

  checkpoint_mngr = tf.train.Checkpoint(model=model)
  checkpoint_mngr.restore(checkpoint_path).assert_existing_objects_matched()
  logging.info('Restored checkpoint: %s', checkpoint_path)

  min_ades = []
  min_fdes = []
  prediction_start_index = 8
  
  for input_batch in tqdm.tqdm(dataset.batch(1)):
    if input_batch['agents/position'].shape[1] == 0:
      continue
      



    positions = np.zeros((1, 1, 20, 2), dtype=np.float32)
    start_x, start_y = 2.0, 2.0

    for i in range(7):
      positions[0, 0, i, 0] = start_x + 0.2 * i  # Update x-coordinate
      positions[0, 0, i, 1] = start_y  # Keep y-coordinate at 2.0


    last_x = positions[0, 0, 6, 0]
    last_y = positions[0, 0, 6, 1]
    positions[0, 0, 7:20, 0] = last_x  # Fill remaining x-coordinates
    positions[0, 0, 7:20, 1] = last_y  # Fill remaining y-coordinates

    print(positions)
    positions = tf.convert_to_tensor(positions)
    input_batch['agents/position'] = positions
      
      
    target = tf.identity(input_batch['agents/position'])
    target_np = input_batch['agents/position'].numpy()
    target_np[:, :, prediction_start_index:] = np.nan
    input_batch['agents/position'] = tf.convert_to_tensor(target_np)
    full_pred, _ = model(input_batch, training=False)
    samples = full_pred['agents/position']
    
    ml_indices = tf.squeeze(
        tf.math.argmax(full_pred['mixture_logits'], axis=-1)
    )
    pred = full_pred['agents/position'][..., ml_indices, :]
    
    
    print(input_batch['agents/position'].shape) 


    for a in range(input_batch['agents/position'].shape[1]): # iterating over agents
      if not tf.reduce_any(tf.math.is_nan(target[0, a])): # prediction will only be reasonable when target has non nan. 
       
       	
        import matplotlib.pyplot as plt
        plt.figure(figsize=(15,10))
        plt.scatter(pred[0, a, prediction_start_index:, 0], pred[0, a, prediction_start_index:, 1], c='blue')
        plt.scatter(target[0, a, prediction_start_index:, 0], target[0, a, prediction_start_index:, 1], c='red')
        plt.scatter(input_batch['agents/position'][0, a, :, 0], input_batch['agents/position'][0, a, :, 1], c='green')
        plt.xlim([-0.5,7.0]) # [-0.5,3.0]
        plt.ylim([-2.5, 4.0]) # [-2.5, 1.0]
        plt.show()
    	    
        
        #min_ades.append(min_ade(samples[0, a, 8:], target[0, a, 8:]))
        #min_fdes.append(min_fde(samples[0, a, 8:], target[0, a, 8:]))
       
    break
    
  #print(f'MinADE: {np.nanmean(min_ades):.2f}')
  #print(f'MinFDE: {np.nanmean(min_fdes):.2f}')


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError('Too many command-line arguments.')

  gin.parse_config_files_and_bindings(
      [os.path.join(_MODEL_PATH.value, 'params', 'operative_config.gin')],
      None,
      skip_unknown=True)
  print('Actual gin config used:')
  print(gin.config_str())

  evaluation(_CKPT_PATH.value)


if __name__ == '__main__':
  logging.set_verbosity(logging.ERROR)
  app.run(main)
