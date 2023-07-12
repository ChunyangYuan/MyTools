import requests
r = requests.post(
    "https://api.deepai.org/api/torch-srgan",
    files={
        'image': open(r'F:\dataset\SIRSTdevkit-master\Misc\Misc_1.png', 'rb'),
    },
    headers={'api-key': 'f32ca362-a6d7-4bb1-9044-9f7d62871cf4'}
)
print(r.json())
