"""Worked values and tests for the RUNTIME vectors and matrices lessons."""
import unittest
from math_model import dot, linear

W = [[2, -1], [-1, 2], [1, 0]]
B = [1, -1]

def features(window):
    if not isinstance(window, str) or len(window) != 3:
        raise ValueError('This teaching extractor expects exactly three characters')
    return [window.count(c) for c in ['a', 't', ' ']]

def batch(rows, weights, bias):
    return [linear(row, weights, bias) for row in rows]

def example():
    x=features('aat'); other=features('a t')
    columns=list(map(list,zip(*W)))
    return {'scope':'Hand-chosen affine scorer; no training; not probabilities',
            'feature_labels':['count a','count t','count SPACE'],
            'output_labels':['t','n'], 'windows':['aat','a t'],
            'x':x,'other':other,'weights':W,'columns':columns,'bias':B,
            'products':[[v*w for v,w in zip(x,col)] for col in columns],
            'sums':[dot(x,col) for col in columns],
            'output':linear(x,W,B),'other_output':linear(other,W,B),
            'batch_inputs':[x,other],'batch_outputs':batch([x,other],W,B),
            'zero_output':linear([0,0,0],W,B),
            'probe':{'input':[3,1,0],'sum':dot([3,1,0],columns[0]),
                     'label':'Arithmetic probe only: counts exceed a three-character window'}}

class MathChecks(unittest.TestCase):
    def test_worked_values(self):
        e=example()
        self.assertEqual(e['products'],[[4,-1,0],[-2,2,0]])
        self.assertEqual(e['sums'],[3,0]);self.assertEqual(e['output'],[4,-1])
        self.assertEqual(e['batch_outputs'],[[4,-1],[3,0]])
    def test_input_order_loss(self):
        self.assertEqual(features('aat'),features('ata'))
        self.assertEqual(features('a t'),[1,1,1])
        with self.assertRaises(ValueError):features('aaaa')
    def test_zero_input_bias(self):
        self.assertEqual(linear([0,0,0],W,B),B)
    def test_isolated_change(self):
        self.assertEqual(dot([3,1,0],[2,-1,1])-dot([2,1,0],[2,-1,1]),2)
    def test_independent_rows(self):
        first=linear([2,1,0],W,B)
        self.assertEqual(batch([[2,1,0],[0,0,0]],W,B),[first,B])
    def test_code_storage_transpose(self):
        stored=list(zip(*W))  # Common Linear storage: [out_features,in_features].
        self.assertEqual([dot([2,1,0],r)+b for r,b in zip(stored,B)],linear([2,1,0],W,B))
    def test_fail_closed_shapes(self):
        for call in [lambda:dot([1,2],[1]),lambda:linear([1,2],W,B),
                     lambda:linear([1,2,3],W,[1]),lambda:linear([1,2],[[1],[2,3]],[1])]:
            with self.assertRaises(ValueError):call()
    def test_invalid_numbers(self):
        for bad in [float('nan'),float('inf'),True,'1']:
            with self.assertRaises(ValueError):dot([bad],[1])
    def test_bias_is_affine(self):
        a=linear([1,0,0],W,B);c=linear([0,1,0],W,B)
        together=linear([1,1,0],W,B)
        self.assertEqual(together,[u+v-b for u,v,b in zip(a,c,B)])

if __name__ == '__main__':
    unittest.main()
