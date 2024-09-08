import os
import json

def read_json_file(file_path):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

def pad(arr, n):
  if len(arr) < n:
    for i in range(n-len(arr)):
      arr.append([])
  for i in range(n):
    if len(arr[i]) < n:
      for j in range(n-len(arr[i])):
        arr[i].append(0)
  return arr

def output_color_remove(output_grid, color):
  for i in range(30):
    for j in range(30):
      if output_grid[i][j] == color:
        output_grid[i][j] = 0

base_folder = 'Old_Filter_Model/Research/concept_data'

folder_list = os.listdir(base_folder)

for folder_name in folder_list:
    folder_path = os.path.join(base_folder, folder_name)

    json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]

    data_by_folder = {'input_data': [], 'output_data': [], 'input_size': [], 'output_size': []}

    for json_file in json_files:
        json_file_path = os.path.join(folder_path, json_file)
        data = read_json_file(json_file_path)

        train_data = data['train']
        for item in train_data:
            data_by_folder['input_data'].append(item['input'])
            data_by_folder['output_data'].append(item['output'])
            data_by_folder['input_size'].append([len(item['input']), len(item['input'][0])])
            data_by_folder['output_size'].append([len(item['output']), len(item['output'][0])])

        test_data = data['test']
        for item in test_data:
            data_by_folder['input_data'].append(item['input'])
            data_by_folder['output_data'].append(item['output'])
            data_by_folder['input_size'].append([len(item['input']), len(item['input'][0])])
            data_by_folder['output_size'].append([len(item['output']), len(item['output'][0])])

    output_file_path = os.path.join('Old_Filter_Model/Research/data', f'{folder_name}.json')

    with open(output_file_path, 'w') as output_file:
        json.dump(data_by_folder, output_file)

json_file_path = 'Old_Filter_Model/Research/data/AboveBelow.json'

# json 파일 읽어오기
with open(json_file_path, 'r') as json_file:
    data = json.load(json_file)

# 데이터 수정 (예시: input_data, output_data에 padding을 적용)
AboveBelow_input_data = data.get("input_data")
AboveBelow_output_data = data.get("output_data")
AboveBelow_input_size = data.get("input_size")
AboveBelow_output_size = data.get("output_size")

for i in range(len(AboveBelow_input_data)):
    AboveBelow_input_data[i] = pad(AboveBelow_input_data[i], 30)
    AboveBelow_output_data[i] = pad(AboveBelow_output_data[i], 30)

# 수정된 데이터를 json 파일에 다시 저장
data["input_data"] = AboveBelow_input_data
data["output_data"] = AboveBelow_output_data
# 다른 필요한 수정 작업 수행

# 수정된 데이터를 저장할 json 파일 경로 (원하는 경로로 설정)
output_json_file_path = 'Old_Filter_Model/Research/data/AboveBelow.json'

# json 파일에 수정된 데이터 저장
with open(output_json_file_path, 'w') as output_json_file:
    json.dump(data, output_json_file)

""" print(AboveBelow_input_data)
print(AboveBelow_output_data)
print(AboveBelow_input_size)
print(AboveBelow_output_size)
 """
# json 파일 경로
json_file_path = 'Old_Filter_Model/Research/data/Center.json'

# json 파일 읽어오기
with open(json_file_path, 'r') as json_file:
    data = json.load(json_file)

# 데이터 수정 (예시: input_data, output_data에 padding을 적용)
Center_input_data = data.get("input_data")
Center_output_data = data.get("output_data")
Center_input_size = data.get("input_size")
Center_output_size = data.get("output_size")

for i in range(len(Center_input_data)):
    Center_input_data[i] = pad(Center_input_data[i], 30)
    Center_output_data[i] = pad(Center_output_data[i], 30)

# 수정된 데이터를 json 파일에 다시 저장
data["input_data"] = Center_input_data
data["output_data"] = Center_output_data
# 다른 필요한 수정 작업 수행

# 수정된 데이터를 저장할 json 파일 경로 (원하는 경로로 설정)
output_json_file_path = 'Old_Filter_Model/Research/data/Center.json'

# json 파일에 수정된 데이터 저장
with open(output_json_file_path, 'w') as output_json_file:
    json.dump(data, output_json_file)

""" print(Center_input_data)
print(Center_output_data)
print(Center_input_size)
print(Center_output_size)
 """
json_file_path = 'Old_Filter_Model/Research/data/CleanUp.json'

# json 파일 읽어오기
with open(json_file_path, 'r') as json_file:
    data = json.load(json_file)

# 데이터 수정 (예시: input_data, output_data에 padding을 적용)
CleanUp_input_data = data.get("input_data")
CleanUp_output_data = data.get("output_data")
CleanUp_input_size = data.get("input_size")
CleanUp_output_size = data.get("output_size")

for i in range(len(CleanUp_input_data)):
    CleanUp_input_data[i] = pad(CleanUp_input_data[i], 30)
    CleanUp_output_data[i] = pad(CleanUp_output_data[i], 30)

# 수정된 데이터를 json 파일에 다시 저장
data["input_data"] = CleanUp_input_data
data["output_data"] = CleanUp_output_data
# 다른 필요한 수정 작업 수행

# 수정된 데이터를 저장할 json 파일 경로 (원하는 경로로 설정)
output_json_file_path = 'Old_Filter_Model/Research/data/CleanUp.json'

# json 파일에 수정된 데이터 저장
with open(output_json_file_path, 'w') as output_json_file:
    json.dump(data, output_json_file)


""" print(CleanUp_input_data)
print(CleanUp_output_data)
print(CleanUp_input_size)
print(CleanUp_output_size)
 """
json_file_path = 'Old_Filter_Model/Research/data/CompleteShape.json'

# json 파일 읽어오기
with open(json_file_path, 'r') as json_file:
    data = json.load(json_file)

# 데이터 수정 (예시: input_data, output_data에 padding을 적용)
CompleteShape_input_data = data.get("input_data")
CompleteShape_output_data = data.get("output_data")
CompleteShape_input_size = data.get("input_size")
CompleteShape_output_size = data.get("output_size")

for i in range(len(CompleteShape_input_data)):
    CompleteShape_input_data[i] = pad(CompleteShape_input_data[i], 30)
    CompleteShape_output_data[i] = pad(CompleteShape_output_data[i], 30)

# 수정된 데이터를 json 파일에 다시 저장
data["input_data"] = CompleteShape_input_data
data["output_data"] = CompleteShape_output_data
# 다른 필요한 수정 작업 수행

# 수정된 데이터를 저장할 json 파일 경로 (원하는 경로로 설정)
output_json_file_path = 'Old_Filter_Model/Research/data/CompleteShape.json'

# json 파일에 수정된 데이터 저장
with open(output_json_file_path, 'w') as output_json_file:
    json.dump(data, output_json_file)

""" print(CompleteShape_input_data)
print(CompleteShape_output_data)
print(CompleteShape_input_size)
print(CompleteShape_output_size)
 """
json_file_path = 'Old_Filter_Model/Research/data/Copy.json'

# json 파일 읽어오기
with open(json_file_path, 'r') as json_file:
    data = json.load(json_file)

# 데이터 수정 (예시: input_data, output_data에 padding을 적용)
Copy_input_data = data.get("input_data")
Copy_output_data = data.get("output_data")
Copy_input_size = data.get("input_size")
Copy_output_size = data.get("output_size")

for i in range(len(Copy_input_data)):
    Copy_input_data[i] = pad(Copy_input_data[i], 30)
    Copy_output_data[i] = pad(Copy_output_data[i], 30)

# 수정된 데이터를 json 파일에 다시 저장
data["input_data"] = Copy_input_data
data["output_data"] = Copy_output_data
# 다른 필요한 수정 작업 수행

# 수정된 데이터를 저장할 json 파일 경로 (원하는 경로로 설정)
output_json_file_path = 'Old_Filter_Model/Research/data/Copy.json'

# json 파일에 수정된 데이터 저장
with open(output_json_file_path, 'w') as output_json_file:
    json.dump(data, output_json_file)

""" print(Copy_input_data)
print(Copy_output_data)
print(Copy_input_size)
print(Copy_output_size)
 """
json_file_path = 'Old_Filter_Model/Research/data/Count.json'

# json 파일 읽어오기
with open(json_file_path, 'r') as json_file:
    data = json.load(json_file)

# 데이터 수정 (예시: input_data, output_data에 padding을 적용)
Count_input_data = data.get("input_data")
Count_output_data = data.get("output_data")
Count_input_size = data.get("input_size")
Count_output_size = data.get("output_size")

for i in range(len(Count_input_data)):
    Count_input_data[i] = pad(Count_input_data[i], 30)
    Count_output_data[i] = pad(Count_output_data[i], 30)

# 수정된 데이터를 json 파일에 다시 저장
data["input_data"] = Count_input_data
data["output_data"] = Count_output_data
# 다른 필요한 수정 작업 수행

# 수정된 데이터를 저장할 json 파일 경로 (원하는 경로로 설정)
output_json_file_path = 'Old_Filter_Model/Research/data/Count.json'

# json 파일에 수정된 데이터 저장
with open(output_json_file_path, 'w') as output_json_file:
    json.dump(data, output_json_file)

json_file_path = 'Old_Filter_Model/Research/data/ExtendToBoundary.json'

# json 파일 읽어오기
with open(json_file_path, 'r') as json_file:
    data = json.load(json_file)

# 데이터 수정 (예시: input_data, output_data에 padding을 적용)
ExtendToBoundary_input_data = data.get("input_data")
ExtendToBoundary_output_data = data.get("output_data")
ExtendToBoundary_input_size = data.get("input_size")
ExtendToBoundary_output_size = data.get("output_size")

for i in range(len(ExtendToBoundary_input_data)):
    ExtendToBoundary_input_data[i] = pad(ExtendToBoundary_input_data[i], 30)
    ExtendToBoundary_output_data[i] = pad(ExtendToBoundary_output_data[i], 30)

# 수정된 데이터를 json 파일에 다시 저장
data["input_data"] = ExtendToBoundary_input_data
data["output_data"] = ExtendToBoundary_output_data
# 다른 필요한 수정 작업 수행

# 수정된 데이터를 저장할 json 파일 경로 (원하는 경로로 설정)
output_json_file_path = 'Old_Filter_Model/Research/data/ExtendToBoundary.json'

# json 파일에 수정된 데이터 저장
with open(output_json_file_path, 'w') as output_json_file:
    json.dump(data, output_json_file)

""" print(ExtendToBoundary_input_data)
print(ExtendToBoundary_output_data)
print(ExtendToBoundary_input_size)
print(ExtendToBoundary_output_size) """

json_file_path = 'Old_Filter_Model/Research/data/ExtractObjects.json'

# json 파일 읽어오기
with open(json_file_path, 'r') as json_file:
    data = json.load(json_file)

# 데이터 수정 (예시: input_data, output_data에 padding을 적용)
ExtractObjects_input_data = data.get("input_data")
ExtractObjects_output_data = data.get("output_data")
ExtractObjects_input_size = data.get("input_size")
ExtractObjects_output_size = data.get("output_size")

for i in range(len(ExtractObjects_input_data)):
    ExtractObjects_input_data[i] = pad(ExtractObjects_input_data[i], 30)
    ExtractObjects_output_data[i] = pad(ExtractObjects_output_data[i], 30)

# 수정된 데이터를 json 파일에 다시 저장
data["input_data"] = ExtractObjects_input_data
data["output_data"] = ExtractObjects_output_data
# 다른 필요한 수정 작업 수행

# 수정된 데이터를 저장할 json 파일 경로 (원하는 경로로 설정)
output_json_file_path = 'Old_Filter_Model/Research/data/ExtractObjects.json'

# json 파일에 수정된 데이터 저장
with open(output_json_file_path, 'w') as output_json_file:
    json.dump(data, output_json_file)

""" print(ExtractObjects_input_data)
print(ExtractObjects_output_data)
print(ExtractObjects_input_size)
print(ExtractObjects_output_size)
 """
json_file_path = 'Old_Filter_Model/Research/data/FilledNotFilled.json'

# json 파일 읽어오기
with open(json_file_path, 'r') as json_file:
    data = json.load(json_file)

# 데이터 수정 (예시: input_data, output_data에 padding을 적용)
FilledNotFilled_input_data = data.get("input_data")
FilledNotFilled_output_data = data.get("output_data")
FilledNotFilled_input_size = data.get("input_size")
FilledNotFilled_output_size = data.get("output_size")

for i in range(len(FilledNotFilled_input_data)):
    FilledNotFilled_input_data[i] = pad(FilledNotFilled_input_data[i], 30)
    FilledNotFilled_output_data[i] = pad(FilledNotFilled_output_data[i], 30)

# 수정된 데이터를 json 파일에 다시 저장
data["input_data"] = FilledNotFilled_input_data
data["output_data"] = FilledNotFilled_output_data
# 다른 필요한 수정 작업 수행

# 수정된 데이터를 저장할 json 파일 경로 (원하는 경로로 설정)
output_json_file_path = 'Old_Filter_Model/Research/data/FilledNotFilled.json'

# json 파일에 수정된 데이터 저장
with open(output_json_file_path, 'w') as output_json_file:
    json.dump(data, output_json_file)

""" print(FilledNotFilled_input_data)
print(FilledNotFilled_output_data)
print(FilledNotFilled_input_size)
print(FilledNotFilled_output_size)
 """
json_file_path = 'Old_Filter_Model/Research/data/HorizontalVertical.json'

# json 파일 읽어오기
with open(json_file_path, 'r') as json_file:
    data = json.load(json_file)

# 데이터 수정 (예시: input_data, output_data에 padding을 적용)
HorizontalVertical_input_data = data.get("input_data")
HorizontalVertical_output_data = data.get("output_data")
HorizontalVertical_input_size = data.get("input_size")
HorizontalVertical_output_size = data.get("output_size")

for i in range(len(HorizontalVertical_input_data)):
    HorizontalVertical_input_data[i] = pad(HorizontalVertical_input_data[i], 30)
    HorizontalVertical_output_data[i] = pad(HorizontalVertical_output_data[i], 30)

# 수정된 데이터를 json 파일에 다시 저장
data["input_data"] = HorizontalVertical_input_data
data["output_data"] = HorizontalVertical_output_data
# 다른 필요한 수정 작업 수행

# 수정된 데이터를 저장할 json 파일 경로 (원하는 경로로 설정)
output_json_file_path = 'Old_Filter_Model/Research/data/HorizontalVertical.json'

# json 파일에 수정된 데이터 저장
with open(output_json_file_path, 'w') as output_json_file:
    json.dump(data, output_json_file)

""" print(HorizontalVertical_input_data)
print(HorizontalVertical_output_data)
print(HorizontalVertical_input_size)
print(HorizontalVertical_output_size)
 """
json_file_path = 'Old_Filter_Model/Research/data/InsideOutside.json'

# json 파일 읽어오기
with open(json_file_path, 'r') as json_file:
    data = json.load(json_file)

# 데이터 수정 (예시: input_data, output_data에 padding을 적용)
InsideOutside_input_data = data.get("input_data")
InsideOutside_output_data = data.get("output_data")
InsideOutside_input_size = data.get("input_size")
InsideOutside_output_size = data.get("output_size")

for i in range(len(InsideOutside_input_data)):
    InsideOutside_input_data[i] = pad(InsideOutside_input_data[i], 30)
    InsideOutside_output_data[i] = pad(InsideOutside_output_data[i], 30)

# 수정된 데이터를 json 파일에 다시 저장
data["input_data"] = InsideOutside_input_data
data["output_data"] = InsideOutside_output_data
# 다른 필요한 수정 작업 수행

# 수정된 데이터를 저장할 json 파일 경로 (원하는 경로로 설정)
output_json_file_path = 'Old_Filter_Model/Research/data/InsideOutside.json'

# json 파일에 수정된 데이터 저장
with open(output_json_file_path, 'w') as output_json_file:
    json.dump(data, output_json_file)

""" print(InsideOutside_input_data)
print(InsideOutside_output_data)
print(InsideOutside_input_size)
print(InsideOutside_output_size)
 """
json_file_path = 'Old_Filter_Model/Research/data/MoveToBoundary.json'

# json 파일 읽어오기
with open(json_file_path, 'r') as json_file:
    data = json.load(json_file)

# 데이터 수정 (예시: input_data, output_data에 padding을 적용)
MoveToBoundary_input_data = data.get("input_data")
MoveToBoundary_output_data = data.get("output_data")
MoveToBoundary_input_size = data.get("input_size")
MoveToBoundary_output_size = data.get("output_size")

for i in range(len(MoveToBoundary_input_data)):
    MoveToBoundary_input_data[i] = pad(MoveToBoundary_input_data[i], 30)
    MoveToBoundary_output_data[i] = pad(MoveToBoundary_output_data[i], 30)

# 수정된 데이터를 json 파일에 다시 저장
data["input_data"] = MoveToBoundary_input_data
data["output_data"] = MoveToBoundary_output_data
# 다른 필요한 수정 작업 수행

# 수정된 데이터를 저장할 json 파일 경로 (원하는 경로로 설정)
output_json_file_path = 'Old_Filter_Model/Research/data/MoveToBoundary.json'

# json 파일에 수정된 데이터 저장
with open(output_json_file_path, 'w') as output_json_file:
    json.dump(data, output_json_file)

""" print(MoveToBoundary_input_data)
print(MoveToBoundary_output_data)
print(MoveToBoundary_input_size)
print(MoveToBoundary_output_size)
 """
json_file_path = 'Old_Filter_Model/Research/data/Order.json'

# json 파일 읽어오기
with open(json_file_path, 'r') as json_file:
    data = json.load(json_file)

# 데이터 수정 (예시: input_data, output_data에 padding을 적용)
Order_input_data = data.get("input_data")
Order_output_data = data.get("output_data")
Order_input_size = data.get("input_size")
Order_output_size = data.get("output_size")

for i in range(len(Order_input_data)):
    Order_input_data[i] = pad(Order_input_data[i], 30)
    Order_output_data[i] = pad(Order_output_data[i], 30)

# 수정된 데이터를 json 파일에 다시 저장
data["input_data"] = Order_input_data
data["output_data"] = Order_output_data
# 다른 필요한 수정 작업 수행

# 수정된 데이터를 저장할 json 파일 경로 (원하는 경로로 설정)
output_json_file_path = 'Old_Filter_Model/Research/data/Order.json'

# json 파일에 수정된 데이터 저장
with open(output_json_file_path, 'w') as output_json_file:
    json.dump(data, output_json_file)

""" 
print(Order_input_data)
print(Order_output_data)
print(Order_input_size)
print(Order_output_size)
 """
json_file_path = 'Old_Filter_Model/Research/data/SameDifferent.json'

# json 파일 읽어오기
with open(json_file_path, 'r') as json_file:
    data = json.load(json_file)

# 데이터 수정 (예시: input_data, output_data에 padding을 적용)
SameDifferent_input_data = data.get("input_data")
SameDifferent_output_data = data.get("output_data")
SameDifferent_input_size = data.get("input_size")
SameDifferent_output_size = data.get("output_size")

for i in range(len(SameDifferent_input_data)):
    SameDifferent_input_data[i] = pad(SameDifferent_input_data[i], 30)
    SameDifferent_output_data[i] = pad(SameDifferent_output_data[i], 30)

# 수정된 데이터를 json 파일에 다시 저장
data["input_data"] = SameDifferent_input_data
data["output_data"] = SameDifferent_output_data
# 다른 필요한 수정 작업 수행

# 수정된 데이터를 저장할 json 파일 경로 (원하는 경로로 설정)
output_json_file_path = 'Old_Filter_Model/Research/data/SameDifferent.json'

# json 파일에 수정된 데이터 저장
with open(output_json_file_path, 'w') as output_json_file:
    json.dump(data, output_json_file)

""" print(SameDifferent_input_data)
print(SameDifferent_output_data)
print(SameDifferent_input_size)
print(SameDifferent_output_size)
 """
json_files = [
    'Old_Filter_Model/Research/data/AboveBelow.json',
    'Old_Filter_Model/Research/data/Center.json',
    'Old_Filter_Model/Research/data/CleanUp.json',
    'Old_Filter_Model/Research/data/CompleteShape.json',
    'Old_Filter_Model/Research/data/Copy.json',
    'Old_Filter_Model/Research/data/Count.json',
    'Old_Filter_Model/Research/data/ExtendToBoundary.json',
    'Old_Filter_Model/Research/data/ExtractObjects.json',
    'Old_Filter_Model/Research/data/FilledNotFilled.json',
    'Old_Filter_Model/Research/data/HorizontalVertical.json',
    'Old_Filter_Model/Research/data/InsideOutside.json',
    'Old_Filter_Model/Research/data/MoveToBoundary.json',
    'Old_Filter_Model/Research/data/Order.json',
    'Old_Filter_Model/Research/data/SameDifferent.json'
]

combined_data = {"input": [], "output": [], "input_size": [], "output_size": [], "task": []}

for json_file_path in json_files:
    task_name = os.path.splitext(os.path.basename(json_file_path))[0]

    with open(json_file_path, 'r') as file:
        data = json.load(file)

    combined_data["input"].append(data['input_data'])
    combined_data["output"].append(data['output_data'])
    combined_data["input_size"].append(data['input_size'])
    combined_data["output_size"].append(data['output_size'])
    combined_data["task"].append(task_name)

concept_true = {"input": [], "output": [], "input_size": [], "output_size": [], "task": []}

for i in range(len(combined_data["task"])):
  for j in range(0, len(combined_data['input'][i]), 2):
    concept_true["input"].append(combined_data['input'][i][j])
    concept_true["output"].append(combined_data['output'][i][j])
    concept_true["input_size"].append(combined_data['input_size'][i][j])
    concept_true["output_size"].append(combined_data['output_size'][i][j])
    concept_true["task"].append(1)

""" print(concept_true["input"])
print(concept_true["output"])
print(concept_true["input_size"])
print(concept_true["output_size"])
print(concept_true["task"])
"""
print(len(concept_true["input"]))
print(len(concept_true["output"]))
print(len(concept_true["input_size"]))
print(len(concept_true["output_size"]))
print(len(concept_true["task"]))
 
# 최종 결과를 JSON 파일로 저장
output_file_path = 'Old_Filter_Model/Research/data/Concept_True.json'
with open(output_file_path, 'w') as output_file:
    json.dump(concept_true, output_file)

    import random

json_file_path = 'Old_Filter_Model/Research/data/Concept_True.json'

concept_false = {"input": [], "output": [], "input_size": [], "output_size": [], "task": []}

with open(json_file_path, 'r') as json_file:
  data = json.load(json_file)

for j in range(0, 28):
  for i in range(0, len(data["task"])-5*j, 25):
    concept_false["input"].append(data["input"][i])
    concept_false["output"].append(data["output"][i+5*j])
    concept_false["input_size"].append(data["input_size"][i])
    concept_false["output_size"].append(data["output_size"][i+5*j])
    concept_false["task"].append(0)

""" for i in range(0, len(data["task"]), 2):
  concept_false["input"].append(data["input"][i])
  concept_false["output"].append(output_color_remove(data["output"][i], random.randint(0, 9)))
  concept_false["input_size"].append(data["input_size"][i])
  concept_false["output_size"].append(data["output_size"][i])
  concept_false["task"].append(0)
 """

""" print(concept_false["input"])
print(concept_false["output"])
print(concept_false["input_size"])
print(concept_false["output_size"])
print(concept_false["task"])
"""
print(len(concept_false["input"]))
print(len(concept_false["output"]))
print(len(concept_false["input_size"]))
print(len(concept_false["output_size"]))
print(len(concept_false["task"]))


# 최종 결과를 JSON 파일로 저장
output_file_path = 'Old_Filter_Model/Research/data/Concept_False.json'
with open(output_file_path, 'w') as output_file:
    json.dump(concept_false, output_file)

json_file_path = 'Old_Filter_Model/Research/data/Concept_False.json'
add_json_path = 'Old_Filter_Model/Research/data/Concept_True.json'

concept_tf = {"input": [], "output": [], "input_size": [], "output_size": [], "task": []}

with open(json_file_path, 'r') as json_file:
  data = json.load(json_file)

with open(add_json_path, 'r') as json_file:
  add_data = json.load(json_file)

concept_tf["input"] = data["input"]
concept_tf["output"] = data["output"]
concept_tf["input_size"] = data["input_size"]
concept_tf["output_size"] = data["output_size"]
concept_tf["task"] = data["task"]

for i in range(len(add_data["input"])):
  concept_tf["input"].insert(int(2.15*i), add_data["input"][i])
  concept_tf["output"].insert(int(2.15*i), add_data["output"][i])
  concept_tf["input_size"].insert(int(2.15*i), add_data["input_size"][i])
  concept_tf["output_size"].insert(int(2.15*i), add_data["output_size"][i])
  concept_tf["task"].insert(int(2.15*i), add_data["task"][i])

""" print(concept_tf["input"])
print(concept_tf["output"])
print(concept_tf["input_size"])
print(concept_tf["output_size"])
print(concept_tf["task"])

print(len(concept_tf["input"]))
print(len(concept_tf["output"]))
print(len(concept_tf["input_size"]))
print(len(concept_tf["output_size"]))
print(len(concept_tf["task"]))
 """

# 최종 결과를 JSON 파일로 저장
output_file_path = 'Old_Filter_Model/Research/data/Concept_TF.json'
with open(output_file_path, 'w') as output_file:
    json.dump(concept_tf, output_file)

json_file_path = 'Old_Filter_Model/Research/data/Concept_TF.json'

with open(json_file_path, 'r') as json_file:
    data = json.load(json_file)

# 데이터 길이 계산
total_len = len(data['input'])
train_len = int(total_len * 0.85)
valid_len = total_len - train_len

train_concept = {"input": [], "output": [], "input_size": [], "output_size": [], "task": []}
valid_concept = {"input": [], "output": [], "input_size": [], "output_size": [], "task": []}

# 데이터 분할

train_concept['input'] = data['input'][:train_len]
train_concept['output'] = data['output'][:train_len]
train_concept['input_size'] = data['input_size'][:train_len]
train_concept['output_size'] = data['output_size'][:train_len]
train_concept['task'] = data['task'][:train_len]

valid_concept['input'] = data['input'][train_len:]
valid_concept['output'] = data['output'][train_len:]
valid_concept['input_size'] = data['input_size'][train_len:]
valid_concept['output_size'] = data['output_size'][train_len:]
valid_concept['task'] = data['task'][train_len:]

""" print(train_concept["input"])
print(train_concept["output"])
print(train_concept["input_size"])
print(train_concept["output_size"])
print(train_concept["task"])
print(valid_concept["input"])
print(valid_concept["output"])
print(valid_concept["input_size"])
print(valid_concept["output_size"])
print(valid_concept["task"])
 """
output_file_path = 'Old_Filter_Model/Research/data/Train_Concept.json'
with open(output_file_path, 'w') as output_file:
    json.dump(train_concept, output_file)

output_file_path = 'Old_Filter_Model/Research/data/Valid_Concept.json'
with open(output_file_path, 'w') as output_file:
    json.dump(valid_concept, output_file)