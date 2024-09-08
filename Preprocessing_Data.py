import os
import json
import numpy as np
import random

def fixed_shuffle_list(input_list, seed=32):
    shuffled_list = input_list.copy()
    random.seed(seed)
    random.shuffle(shuffled_list)
    return shuffled_list

def output_color_remove(output_grid):
    for i in range(len(output_grid)):
        for j in range(len(output_grid[i])):
            color = np.max(output_grid[i][j])
            for k in range(len(output_grid[i][j])):
                for l in range(len(output_grid[i][j][k])):
                    if output_grid[i][j][k][l] == color:
                        output_grid[i][j][k][l] = 0
    return output_grid
  
def add_noise(output_grid, noise_level=5.0):
    for i in range(len(output_grid)):
        for j in range(len(output_grid[i])):
            coords = np.array(output_grid[i][j])
            noisy_coords = coords + np.random.normal(scale=noise_level, size=coords.shape)
            noisy_coords = np.clip(np.round(noisy_coords), 1, 10)
            output_grid[i][j] = noisy_coords.astype(int)
    return output_grid

def swap_coordinates(output_grid):
    for i in range(len(output_grid)):
        for j in range(len(output_grid[i])):
            coords = output_grid[i][j]
            np.random.shuffle(coords)
            output_grid[i][j] = coords
    return output_grid

def add_noise_task(output_grid, noise_level=5.0):
    for i in range(len(output_grid)):
        coords = np.array(output_grid[i])
        noisy_coords = coords + np.random.normal(scale=noise_level, size=coords.shape)
        noisy_coords = np.clip(np.round(noisy_coords), 1, 10)
        output_grid[i] = noisy_coords.astype(int)
    return output_grid
  
def swap_coordinates_task(coords):
    np.random.shuffle(coords)
    return coords

def color_move(arr3):
  for i in range(len(arr3)):
    for j in range(len(arr3[i])):
      for k in range(len(arr3[i][j])):
        arr3[i][j][k] += 1
  return arr3

def collect_data(dataset):
    inputs = [example['input'] for example in dataset]
    outputs = [example['output'] for example in dataset]
    return inputs, outputs

def read_data_from_json(json_file_path, task):
    try:
        # JSON 파일을 열고 읽기 모드로 엽니다.
        with open(json_file_path, "r") as json_file:
            # JSON 데이터를 읽습니다.
            data = json.load(json_file)

            train_data = data[task]

            return train_data
    except FileNotFoundError:
        print(f"File not found: {json_file_path}")
        return None
    except KeyError:
        print(f"Key 'train' not found in the JSON data.")
        return None
    
def combine_data_from_directory(directory_path, task):
    combined_data = {
        "input": [],
        "output": [],
        "task": []
    }

    for root, dirs, files in os.walk(directory_path):
        for filename in files:
            if filename.endswith(".json"):
                json_file_path = os.path.join(root, filename)
                data = collect_data(read_data_from_json(json_file_path, task))
                if data is not None:
                    combined_data["input"].append(color_move(data[0]))
                    combined_data["output"].append(color_move(data[1]))
                    combined_data["task"].append(os.path.basename(root))

    return combined_data

def pad(arr, n):
  if isinstance(arr, np.ndarray):
    arr = arr.tolist()
  if len(arr) < n:
    for i in range(n-len(arr)):
      arr.append([])
  for i in range(n):
    if len(arr[i]) < n:
      for j in range(n-len(arr[i])):
        arr[i].append(0)
  return arr

def split_data(data, train_ratio):
    total_samples = len(data["input"])
    train_samples = int(total_samples * train_ratio)

    train_data = {
        "input": data["input"][:train_samples],
        "output": data["output"][:train_samples],
        "input_size": data["input_size"][:train_samples],
        "output_size": data["output_size"][:train_samples],
        "task": data["task"][:train_samples],
    }

    valid_data = {
        "input": data["input"][train_samples:],
        "output": data["output"][train_samples:],
        "input_size": data["input_size"][train_samples:],
        "output_size": data["output_size"][train_samples:],
        "task": data["task"][train_samples:],
    }

    return train_data, valid_data

top_folder_path = "/home/jovyan/Desktop/Wongyu/concept_data"
arc_folder_path = "/home/jovyan/Desktop/Wongyu/ARC_data"

combined_arc_data = combine_data_from_directory(arc_folder_path, "train")
combined_train_data = combine_data_from_directory(top_folder_path, "train")
combined_test_data = combine_data_from_directory(top_folder_path, "test")

combined_all_data = {"input": [], "output": [], "input_size": [], "output_size": [], "task": []}
combined_false_data = {"input": [], "output": [], "input_size": [], "output_size": [], "task": []}

combined_all_data['task'].extend(combined_train_data['task'])
combined_all_data['task'].extend(combined_test_data['task'])
combined_all_data['task'].extend(combined_arc_data['task'])

combined_false_data['task'].extend(combined_train_data['task'])
combined_false_data['task'].extend(combined_test_data['task'])

combined_false_data['task'].extend(combined_train_data['task'])
combined_false_data['task'].extend(combined_test_data['task'])

# combined_false_data['task'].extend(combined_train_data['task'])
# combined_false_data['task'].extend(combined_test_data['task'])

#-----------------------Not task data------------------------#

combined_all_data['input'].extend(combined_train_data['input'])
combined_all_data['input'].extend(combined_test_data['input'])

combined_all_data['output'].extend(combined_train_data['output'])
combined_all_data['output'].extend(combined_test_data['output'])

combined_all_data['input'].extend(combined_arc_data['input'])
combined_all_data['output'].extend(combined_arc_data['output'])

#------------------------false data--------------------------#

# combined_false_data['input'].extend(combined_train_data['input'])
# combined_false_data['input'].extend(combined_test_data['input'])

# combined_false_data['output'].extend(output_color_remove(combined_train_data['output']))
# combined_false_data['output'].extend(output_color_remove(combined_test_data['output']))

combined_false_data['input'].extend(combined_train_data['input'])
combined_false_data['input'].extend(combined_test_data['input'])

combined_false_data['output'].extend(combined_train_data['output'])
combined_false_data['output'].extend(combined_test_data['output'])

combined_false_data['input'].extend(combined_train_data['input'])
combined_false_data['input'].extend(combined_test_data['input'])

combined_false_data['output'].extend(combined_train_data['output'])
combined_false_data['output'].extend(combined_test_data['output'])

for i in range(len(combined_false_data['task'])):
  t = combined_false_data['task'][i]
  if t == 'Center':
     combined_false_data['input'][i] = swap_coordinates_task(combined_false_data['input'][i])
  elif t == 'MoveToBoundary':
     combined_false_data['input'][i] = swap_coordinates_task(combined_false_data['input'][i])
  elif t == 'FilledNotFilled':
     combined_false_data['input'][i] = swap_coordinates_task(add_noise_task(combined_false_data['input'][i]))
  elif t == 'InsideOutside':
     combined_false_data['input'][i] = swap_coordinates_task(combined_false_data['input'][i])
  elif t == 'AboveBelow':
     combined_false_data['input'][i] = swap_coordinates_task(combined_false_data['input'][i])
  elif t == 'ExtractObjects':
     combined_false_data['input'][i] = swap_coordinates_task(add_noise_task(combined_false_data['input'][i]))
  elif t == 'Order':
     combined_false_data['input'][i] = swap_coordinates_task(add_noise_task(combined_false_data['input'][i]))
  elif t == 'HorizontalVertical':
     combined_false_data['input'][i] = add_noise_task(add_noise_task(combined_false_data['input'][i]))
  elif t == 'SameDifferent':
     combined_false_data['input'][i] = add_noise_task(add_noise_task(combined_false_data['input'][i]))
  elif t == 'ExtendToBoundary':
     combined_false_data['input'][i] = swap_coordinates_task(combined_false_data['input'][i])
  elif t == 'CleanUp':
     combined_false_data['input'][i] = swap_coordinates_task(combined_false_data['input'][i])
  elif t == 'Copy':
     combined_false_data['input'][i] = add_noise_task(add_noise_task(combined_false_data['input'][i]))
  elif t == 'Count':
     combined_false_data['input'][i] = add_noise_task(add_noise_task(combined_false_data['input'][i]))
  elif t == 'CompleteShape':
     combined_false_data['input'][i] = swap_coordinates_task(add_noise_task(combined_false_data['input'][i]))
  elif t == 'TopBottom2D':
     combined_false_data['input'][i] = swap_coordinates_task(add_noise_task(combined_false_data['input'][i]))
  elif t == 'TopBottom3D':
     combined_false_data['input'][i] = swap_coordinates_task(add_noise_task(combined_false_data['input'][i]))
  else:
     print("something has not been changed.")
#------------------------false data--------------------------#

for i in range(len(combined_all_data['input'])):
  t_temp_input = []
  t_temp_output = []
  for j in range(len(combined_all_data['input'][i])):
    row = len(combined_all_data['input'][i][j])
    column = len(combined_all_data['input'][i][j][0])
    t_temp_input.append([row, column])

    row = len(combined_all_data['output'][i][j])
    column = len(combined_all_data['output'][i][j][0])
    t_temp_output.append([row, column])
    
  combined_all_data['input_size'].append(t_temp_input)
  combined_all_data['output_size'].append(t_temp_output)

for i in range(len(combined_false_data['input'])):
  f_temp_input = []
  f_temp_output = []
  for j in range(len(combined_false_data['input'][i])):
    row = len(combined_false_data['input'][i][j])
    column = len(combined_false_data['input'][i][j][0])
    f_temp_input.append([row, column])

    row = len(combined_false_data['output'][i][j])
    column = len(combined_false_data['output'][i][j][0])
    f_temp_output.append([row, column])

  combined_false_data['input_size'].append(f_temp_input)
  combined_false_data['output_size'].append(f_temp_output)

for i in range(len(combined_all_data['task'])):
  combined_all_data['task'][i] = 1

for i in range(len(combined_false_data['task'])):
  combined_false_data['task'][i] = 0  
  
for i in range(len(combined_false_data['task'])):
  combined_all_data["input"].insert(int(2.16*i), combined_false_data["input"][i])
  combined_all_data["output"].insert(int(2.16*i), combined_false_data["output"][i])
  combined_all_data["input_size"].insert(int(2.16*i), combined_false_data["input_size"][i])
  combined_all_data["output_size"].insert(int(2.16*i), combined_false_data["output_size"][i])
  combined_all_data["task"].insert(int(2.16*i), combined_false_data["task"][i])

for i in range(len(combined_all_data['input'])):
  for j in range(len(combined_all_data['input'][i])):
    combined_all_data['input'][i][j] = pad(combined_all_data['input'][i][j], 30)
    combined_all_data['output'][i][j] = pad(combined_all_data['output'][i][j], 30)

combined_all_data["input"] = fixed_shuffle_list(combined_all_data["input"], 32)
combined_all_data["output"] = fixed_shuffle_list(combined_all_data["output"], 32)
combined_all_data["input_size"] = fixed_shuffle_list(combined_all_data["input_size"], 32)
combined_all_data["output_size"] = fixed_shuffle_list(combined_all_data["output_size"], 32)
combined_all_data["task"] = fixed_shuffle_list(combined_all_data["task"], 32)

output_all_directory = "/home/jovyan/Desktop/Wongyu/multiple_data/all_data_for_filter.json"
output_train_directory = "/home/jovyan/Desktop/Wongyu/multiple_data/train_data_for_filter.json"
output_valid_directory = "/home/jovyan/Desktop/Wongyu/multiple_data/valid_data_for_filter.json"

combined_train_data, combined_valid_data = split_data(combined_all_data, 0.85)

with open(output_all_directory, "w") as json_file:
    json.dump(combined_all_data, json_file)
    
with open(output_train_directory, "w") as json_file:
    json.dump(combined_train_data, json_file)
    
with open(output_valid_directory, "w") as json_file:
    json.dump(combined_valid_data, json_file)