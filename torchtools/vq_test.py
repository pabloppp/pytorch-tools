import torch
import unittest
from vq import VectorQuantize

class VQTest(unittest.TestCase):

	def test_vq_grad(self):
		model = VectorQuantize(8, 16)
	
		pred, pred_w_grad = model(torch.ones_like(model.codebook.weight))
		
		self.assertEqual(pred.requires_grad, False)
		self.assertEqual(pred_w_grad.requires_grad, True)

if __name__ == '__main__':
	unittest.main()