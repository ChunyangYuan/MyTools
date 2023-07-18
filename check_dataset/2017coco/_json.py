import json

data = {
    "employees": [
        {
            "name": "John Doe",
            "age": 30,
            "position": "Manager"
        },
        {
            "name": "Jane Smith",
            "age": 25,
            "position": "Developer"
        }],
    "boss": [
        {
            "name": "Mike Johnson",
            "age": 35,
            "position": "Sales Executive"
        }]
}
# save
with open('employees.json', 'w') as f:
    json.dump(data, f, indent=4)

# read
with open('employees.json', 'r') as f:
    data = json.load(f)

# 输出员工信息
for employee in data['employees']:
    print(f"Name: {employee['name']}")
    print(f"Age: {employee['age']}")
    print(f"Position: {employee['position']}")
    print()
