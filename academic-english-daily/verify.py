#!/usr/bin/env python3
import json

data = json.loads(open('content.json').read())
print('✅ Valid JSON syntax')
print(f'Vocabulary: {len(data["vocabulary"])} items')
print(f'Patterns: {len(data["patterns"])} items')
print(f'Tips: {len(data["tips"])} items')

assert len(data['vocabulary']) == 15, f'Expected 15 vocab, got {len(data["vocabulary"])}'
assert len(data['patterns']) == 5, f'Expected 5 patterns, got {len(data["patterns"])}'
assert len(data['tips']) == 3, f'Expected 3 tips, got {len(data["tips"])}'
print('✅ All counts correct')

# Check math terms
math_terms = ['convex', 'orthogonal', 'eigenspace', 'curvature', 'isomorphic', 'subspace']
vocab_words = [v['word'] for v in data['vocabulary']]
math_found = [t for t in math_terms if t in vocab_words]
print(f'Math-oriented terms: {math_found}')
print(f'Math count: {len(math_found)}')
print('✅ Verification complete')
