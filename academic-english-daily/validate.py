import json
data = json.load(open('content.json'))
print('VALID JSON')
print(f'Day: {data["day"]}')
print(f'Topic: {data["topic"]}')
print(f'Vocab words: {[v["word"] for v in data["vocabulary"]]}')
print(f'Patterns: {len(data["patterns"])}')
print(f'Tips: {len(data["tips"])}')
